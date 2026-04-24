"""
Step 3: Train RotatE KG embeddings using PyKEEN.
Input:  kg/amr_triples.tsv (60K triples, 25K entities, 6 relations)
Output: kg/embeddings/ — entity and relation embedding numpy arrays + model
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from paths import *

import torch
import numpy as np
import json
import logging

assert torch.backends.mps.is_available(), "ABORT: MPS backend not available"
device_name = "mps"
print(f"[device] Using: {device_name}")

# ── Monkey-patch: MPS cannot compute norm on complex tensors ────────────────
# Decompose complex → (real, imag), compute L2 norm manually on MPS.
import pykeen.utils as _pu

_original_negative_norm = _pu.negative_norm

def _mps_safe_negative_norm(x, p=2, power_norm=False):
    if x.is_complex():
        # ||z||_2 = sqrt(real^2 + imag^2)  per-element, then L2 across dim
        x_real = torch.view_as_real(x)            # (..., 2)
        x_sq = (x_real * x_real).sum(dim=-1)      # (...,) per-complex-element
        if p == 2 and not power_norm:
            return -x_sq.sum(dim=-1).sqrt()
        elif power_norm:
            return -x_sq.pow(p / 2).sum(dim=-1)
        else:
            return -x_sq.sqrt().norm(p=p, dim=-1)
    return _original_negative_norm(x, p=p, power_norm=power_norm)

_pu.negative_norm = _mps_safe_negative_norm
# Also patch the module-level reference used by RotatE interaction
import pykeen.nn.modules as _pm
_pm.negative_norm = _mps_safe_negative_norm
print("[patch] Patched PyKEEN negative_norm for MPS complex support")

KG_DIR = os.path.join(PROJECT_DIR, "kg")
EMB_DIR = os.path.join(KG_DIR, "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

t0 = time.time()

# ── 1. Load triples ────────────────────────────────────────────────────────
print("[1/4] Loading triples...")
from pykeen.triples import TriplesFactory
import pandas as pd

triples_path = os.path.join(KG_DIR, "amr_triples.tsv")
tf = TriplesFactory.from_path(
    triples_path,
    create_inverse_triples=False,
)
print(f"  Triples: {tf.num_triples:,}")
print(f"  Entities: {tf.num_entities:,}")
print(f"  Relations: {tf.num_relations:,}")

# Train/validation/test split (80/10/10)
training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)
print(f"  Train: {training.num_triples:,}, Val: {validation.num_triples:,}, Test: {testing.num_triples:,}")

# ── 1.5. Sanity check: 2-epoch MPS test ────────────────────────────────────
print("\n[1.5/4] MPS sanity check (2 epochs)...")
from pykeen.pipeline import pipeline
from pykeen.training import TrainingCallback

result_test = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model="RotatE",
    model_kwargs=dict(embedding_dim=64),
    training_kwargs=dict(num_epochs=2, batch_size=512),
    device=device_name,
    random_seed=42,
)
print("✅ MPS patch verified — 2 epoch test passed")
print(f"  Device used: {next(result_test.model.parameters()).device}")
del result_test

# ── 2. Train RotatE ────────────────────────────────────────────────────────
print("\n[2/4] Training RotatE (embedding_dim=64, 300 epochs)...")

# Epoch-loss logger
log_path = os.path.join(KG_DIR, "rotate_training.log")
class EpochLogger(TrainingCallback):
    def __init__(self):
        super().__init__()
        self._fh = open(log_path, "w")
        self._fh.write("epoch,loss\n")
    def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
        self._fh.write(f"{epoch},{epoch_loss:.6f}\n")
        self._fh.flush()
    def post_train(self, **kwargs):
        self._fh.close()

epoch_logger = EpochLogger()

result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model="RotatE",
    model_kwargs=dict(
        embedding_dim=64,
    ),
    optimizer="Adam",
    optimizer_kwargs=dict(lr=1e-3),
    training_kwargs=dict(
        num_epochs=300,
        batch_size=512,
        callbacks=[epoch_logger],
    ),
    evaluation_kwargs=dict(batch_size=256),
    stopper="early",
    stopper_kwargs=dict(
        patience=20,
        frequency=10,
        metric="hits_at_10",
    ),
    device=device_name,
    random_seed=42,
)

# ── 3. Extract and save embeddings ─────────────────────────────────────────
print("\n[3/4] Extracting embeddings...")
model = result.model
model.eval()

# Entity embeddings
entity_repr = model.entity_representations[0]
entity_emb = entity_repr(indices=None).detach().cpu().numpy()
print(f"  Entity embeddings shape: {entity_emb.shape}")

# Relation embeddings
relation_repr = model.relation_representations[0]
relation_emb = relation_repr(indices=None).detach().cpu().numpy()
print(f"  Relation embeddings shape: {relation_emb.shape}")

# Save as numpy arrays
np.save(os.path.join(EMB_DIR, "entity_embeddings.npy"), entity_emb)
np.save(os.path.join(EMB_DIR, "relation_embeddings.npy"), relation_emb)

# Save entity-to-index and relation-to-index mappings (PyKEEN's own)
entity_to_id_pykeen = training.entity_to_id
relation_to_id_pykeen = training.relation_to_id

with open(os.path.join(EMB_DIR, "pykeen_entity_to_id.json"), "w") as f:
    json.dump(entity_to_id_pykeen, f)
with open(os.path.join(EMB_DIR, "pykeen_relation_to_id.json"), "w") as f:
    json.dump(relation_to_id_pykeen, f)

# Save the full model
torch.save(model.state_dict(), os.path.join(EMB_DIR, "rotate_model.pt"))

# ── 4. Evaluation metrics ──────────────────────────────────────────────────
print("\n[4/4] Evaluation metrics...")
metrics = result.metric_results.to_dict()

# Extract key metrics
key_metrics = {}
for key in ["hits_at_1", "hits_at_3", "hits_at_5", "hits_at_10",
            "mean_rank", "mean_reciprocal_rank",
            "adjusted_mean_rank_index"]:
    # Try both realistic and optimistic
    for side in ["both", "head", "tail"]:
        for mode in ["realistic", "optimistic"]:
            full_key = f"{side}.{mode}.{key}"
            if full_key in metrics:
                if side == "both" and mode == "realistic":
                    key_metrics[key] = metrics[full_key]
                    print(f"  {key}: {metrics[full_key]:.4f}")

summary = {
    "embedding_dim": 64,
    "num_epochs_trained": result.stopper.best_epoch if hasattr(result, 'stopper') and result.stopper else 300,
    "entity_embedding_shape": list(entity_emb.shape),
    "relation_embedding_shape": list(relation_emb.shape),
    "metrics": key_metrics,
    "elapsed_seconds": round(time.time() - t0, 1)
}
with open(os.path.join(EMB_DIR, "rotate_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print(f"  All outputs in: {EMB_DIR}")
print("\nDONE")
