import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import torch
from scipy import sparse
import pytorch_lightning as pl
import torch.nn as nn

FUSED_DIM = 128

class GenomicOnly(pl.LightningModule):
    """k-mer features only, no KG branch."""
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid()
        )
    def forward(self, kmer_x, gene_embeds):
        g = self.encoder(kmer_x)
        logits = self.classifier(g)
        gene_preds = self.gene_head(g)
        dummy_attn = torch.ones(kmer_x.shape[0], gene_embeds.shape[1], device=kmer_x.device) / gene_embeds.shape[1]
        dummy_gate = torch.full((kmer_x.shape[0], FUSED_DIM), 1.0, device=kmer_x.device)
        return logits, gene_preds, dummy_attn, dummy_gate

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ── 1. Load features ───────────────────────────────────────────────
print("[1/3] Loading features and genomic_only model...")
t0 = time.time()

with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)

test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
n_test = len(test_ids)
gene_names = test_data["gene_names"]

KMER_DIM = len(all_features)
NUM_GENES = len(gene_names)

# Load model
import glob
ckpt_files = glob.glob(os.path.join(MODEL_DIR, "checkpoints/ablation_genomic_only/best*.ckpt"))
ckpt_path = max(ckpt_files, key=os.path.getmtime)
print(f"Loading checkpoint: {ckpt_path}")
model = GenomicOnly.load_from_checkpoint(ckpt_path, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
model.eval()
model.to('cpu')

# Load test data
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_sample_indices = [sample_to_idx[s] for s in test_ids]
X_test = X_sparse[test_sample_indices].toarray().astype(np.float32)

# Gene embeds for GenomicOnly (dummy but required by forward pass)
gene_embeds_test = np.zeros((n_test, NUM_GENES, KG_EMBED_DIM), dtype=np.float32)

# ── 2. Compute SHAP ─────────────────────────────────────────────────────────
print("\n[2/3] Computing SHAP via gradient×input approximation for genomic_only...")

X_test_tensor = torch.tensor(X_test, requires_grad=True, dtype=torch.float32)
ge_test_tensor = torch.tensor(gene_embeds_test, dtype=torch.float32)

shap_values = np.zeros_like(X_test)

with torch.enable_grad():
    for i in range(n_test):
        if i % 1000 == 0:
            print(f"    [{i}/{n_test}]...")

        x_i = torch.tensor(X_test[i:i+1], requires_grad=True, dtype=torch.float32)
        ge_i = ge_test_tensor[i:i+1]

        logits, _, _, _ = model(x_i, ge_i)
        logit_r = logits[0, 1]

        logit_r.backward()

        grad = x_i.grad.detach().numpy()[0]
        shap_values[i] = grad * X_test[i]

        x_i.requires_grad_(False)

print(f"  SHAP values computed: shape {shap_values.shape}")

# ── 3. Save raw SHAP ────────────────────────────────────────────────────────
print("\n[3/3] Saving raw SHAP values for baseline alignment metrics...")
np.savez_compressed(
    os.path.join(EXPLAIN_DIR, "shap_raw_genomic_only.npz"),
    shap_values=shap_values,
    feature_names=np.array(all_features),
)

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.1f}s")
print("DONE — shap_genomic_only.py")
