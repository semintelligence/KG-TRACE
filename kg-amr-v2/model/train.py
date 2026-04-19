"""
Step 6: Dataset + Training Pipeline for KG-AMR
- Loads binary mutation matrix (41460 × 17352)
- Maps per-genome mutated genes → RotatE embeddings (26 genes × 64 dims)
- Trains KG-AMR with early stopping on val F1-macro
- Drug: INH (isoniazid) — largest, best-balanced dataset (39.8% R)
"""
import sys, os, time, csv, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "KG-AMR" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# MPS check
assert torch.backends.mps.is_available(), "MPS not available — need Apple Silicon"
DEVICE = "mps"

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

DRUG = "INH"  # isoniazid — best balanced

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/6] Loading data...")
t0 = time.time()

# Mutation matrix
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

print(f"  Mutation matrix: {X_sparse.shape}")

# Labels for INH — deduplicate index (keep first occurrence)
labels_df = pd.read_parquet(os.path.join(FEATURES_DIR, f"labels/labels_{DRUG}.parquet"))
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]
print(f"  {DRUG} labels (dedup): {len(labels_df)} ({(labels_df['label']==1).sum()} R / {(labels_df['label']==0).sum()} S)")

# Find intersection of mutation matrix samples and labeled samples
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
print(f"  Overlapping samples: {len(labeled_samples)}")

# Build aligned arrays
sample_indices = [sample_to_idx[s] for s in labeled_samples]
X = X_sparse[sample_indices].toarray().astype(np.float32)
y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)
sample_ids = np.array(labeled_samples)

print(f"  Final dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Label distribution: {(y==1).sum()} R / {(y==0).sum()} S ({100*(y==1).mean():.1f}% R)")

# ── 2. Build gene embeddings ────────────────────────────────────────────────
print("\n[2/6] Building per-genome gene embeddings...")

# Load KG embeddings (RotatE stores complex64 — use magnitude for real-valued input)
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
if np.iscomplexobj(entity_emb_raw):
    print(f"  RotatE embeddings are complex64 — using magnitude (preserves KG_EMBED_DIM={KG_EMBED_DIM})")
    entity_emb = np.abs(entity_emb_raw).astype(np.float32)
else:
    entity_emb = entity_emb_raw.astype(np.float32)
print(f"  Entity embeddings: {entity_emb.shape}, dtype={entity_emb.dtype}")
with open(os.path.join(KG_DIR, "embeddings/pykeen_entity_to_id.json")) as f:
    entity_to_id = json.load(f)

# Catalogue genes (the 26 WHO genes)
with open(os.path.join(KG_DIR, "gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
CATALOGUE_GENES = sorted(gene_mechanism.keys())
NUM_GENES = len(CATALOGUE_GENES)
print(f"  Catalogue genes: {NUM_GENES}")

# Get embedding index for each catalogue gene
gene_emb_indices = []
for g in CATALOGUE_GENES:
    idx = entity_to_id.get(g)
    assert idx is not None, f"Gene {g} not in entity_to_id!"
    gene_emb_indices.append(idx)

# Gene embedding matrix: [26, 64]
gene_emb_matrix = entity_emb[gene_emb_indices].astype(np.float32)
print(f"  Gene embedding matrix: {gene_emb_matrix.shape}")

# For each feature, determine which gene it belongs to
feature_to_gene_idx = {}
for fi, feat in enumerate(all_features):
    gene_name = feat.split(":")[0]
    if gene_name in CATALOGUE_GENES:
        feature_to_gene_idx[fi] = CATALOGUE_GENES.index(gene_name)

# For each genome, build gene presence vector and gene embeddings
# gene_presence[i, j] = 1 if genome i has at least one mutation in gene j
gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for fi, gi in feature_to_gene_idx.items():
    col = X[:, fi]
    gene_presence[:, gi] = np.maximum(gene_presence[:, gi], col)

n_with_genes = (gene_presence.sum(axis=1) > 0).sum()
n_zero_genes = (gene_presence.sum(axis=1) == 0).sum()
print(f"  Genomes with ≥1 mutated gene: {n_with_genes}")
print(f"  Genomes with 0 mutated genes: {n_zero_genes} (will use zero vectors)")

# Build per-genome gene embeddings: [n_samples, 26, 64]
# If a gene has no mutation in a genome, its embedding is zeroed out
gene_embeds_all = np.zeros((X.shape[0], NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(X.shape[0]):
    for j in range(NUM_GENES):
        if gene_presence[i, j] > 0:
            gene_embeds_all[i, j, :] = gene_emb_matrix[j]

print(f"  Gene embeddings tensor: {gene_embeds_all.shape}")

# ── 3. Train/Val/Test split ─────────────────────────────────────────────────
print("\n[3/6] Stratified 70:15:15 split...")

# Use index-based splitting to handle multiple arrays
indices = np.arange(X.shape[0])
idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
ge_train, ge_val, ge_test = gene_embeds_all[idx_train], gene_embeds_all[idx_val], gene_embeds_all[idx_test]
gp_train, gp_val, gp_test = gene_presence[idx_train], gene_presence[idx_val], gene_presence[idx_test]
sid_train, sid_val, sid_test = sample_ids[idx_train], sample_ids[idx_val], sample_ids[idx_test]

print(f"  Train: {X_train.shape[0]} ({(y_train==1).mean()*100:.1f}% R)")
print(f"  Val:   {X_val.shape[0]} ({(y_val==1).mean()*100:.1f}% R)")
print(f"  Test:  {X_test.shape[0]} ({(y_test==1).mean()*100:.1f}% R)")

# Save split indices for baselines
split_info = {
    "drug": DRUG,
    "train_ids": sid_train.tolist(),
    "val_ids": sid_val.tolist(),
    "test_ids": sid_test.tolist(),
}
split_path = os.path.join(MODEL_DIR, "split_ids.json")
with open(split_path, "w") as f:
    json.dump(split_info, f)
print(f"  Saved split IDs to {split_path}")


# ── 4. PyTorch Dataset ──────────────────────────────────────────────────────
print("\n[4/6] Building datasets & loaders...")

class AMRDataset(Dataset):
    def __init__(self, X, gene_embeds, y_amr, gene_presence):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.gene_embeds = torch.tensor(gene_embeds, dtype=torch.float32)
        self.y_amr = torch.tensor(y_amr, dtype=torch.long)
        self.gene_presence = torch.tensor(gene_presence, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.gene_embeds[idx], self.y_amr[idx], self.gene_presence[idx]

train_ds = AMRDataset(X_train, ge_train, y_train, gp_train)
val_ds = AMRDataset(X_val, ge_val, y_val, gp_val)
test_ds = AMRDataset(X_test, ge_test, y_test, gp_test)

# Use num_workers=0 on MPS to avoid multiprocessing issues
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# ── 5. Model + Training ─────────────────────────────────────────────────────
print("\n[5/6] Training KG-AMR...")
from model.kg_amr import KGAMR

KMER_DIM = X.shape[1]  # 17352
model = KGAMR(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
print(f"  kmer_dim={KMER_DIM}, num_genes={NUM_GENES}")
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")

# Callbacks
early_stop = EarlyStopping(monitor="val_f1_macro", mode="max", patience=10, verbose=True)
checkpoint = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="best_model",
    monitor="val_f1_macro",
    mode="max",
    save_top_k=1,
    verbose=True,
)

# Training log CSV
log_path = os.path.join(MODEL_DIR, "training_log.csv")
log_file = open(log_path, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow(["epoch", "train_loss", "val_f1_macro"])


class LogCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", float("nan"))
        val_f1 = trainer.callback_metrics.get("val_f1_macro", float("nan"))
        epoch = trainer.current_epoch
        tl = train_loss.item() if hasattr(train_loss, "item") else float(train_loss)
        vf = val_f1.item() if hasattr(val_f1, "item") else float(val_f1)
        log_writer.writerow([epoch, f"{tl:.6f}", f"{vf:.6f}"])
        log_file.flush()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d} | train_loss={tl:.4f} | val_f1={vf:.4f}")


trainer = pl.Trainer(
    max_epochs=100,
    accelerator="mps",
    devices=1,
    callbacks=[early_stop, checkpoint, LogCallback()],
    enable_progress_bar=True,
    log_every_n_steps=10,
)

trainer.fit(model, train_loader, val_loader)
log_file.close()

print(f"\n  Best model: {checkpoint.best_model_path}")
print(f"  Best val_f1_macro: {checkpoint.best_model_score:.4f}")
print(f"  Training log: {log_path}")

# ── 6. Test evaluation ──────────────────────────────────────────────────────
print("\n[6/6] Evaluating on test set...")

# Load best checkpoint
best_model = KGAMR.load_from_checkpoint(
    checkpoint.best_model_path,
    kmer_dim=KMER_DIM,
    num_genes=NUM_GENES,
)
best_model.eval()
best_model = best_model.to("cpu")  # run inference on CPU for stability

all_preds = []
all_probs = []
all_labels = []
all_attn = []
all_gate = []

with torch.no_grad():
    for batch in test_loader:
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, attn_weights, gate = best_model(kmer_x, gene_embeds)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.numpy())
        all_probs.append(probs[:, 1].numpy())
        all_labels.append(y_amr.numpy())
        all_attn.append(attn_weights.numpy())
        all_gate.append(gate.numpy())

all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_attn = np.concatenate(all_attn)
all_gate = np.concatenate(all_gate)

# Compute metrics
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

test_f1_macro = f1_score(all_labels, all_preds, average="macro")
test_auroc = roc_auc_score(all_labels, all_probs)
test_precision = precision_score(all_labels, all_preds, average="macro")
test_recall = recall_score(all_labels, all_preds, average="macro")
test_cm = confusion_matrix(all_labels, all_preds)

print(f"\n  === KG-AMR Test Results ({DRUG}) ===")
print(f"  AUROC:     {test_auroc:.4f}")
print(f"  F1-macro:  {test_f1_macro:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  Confusion Matrix:")
print(f"    {test_cm}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['S','R'])}")

# Save test results
test_results = {
    "drug": DRUG,
    "model": "KG-AMR",
    "auroc": float(test_auroc),
    "f1_macro": float(test_f1_macro),
    "precision_macro": float(test_precision),
    "recall_macro": float(test_recall),
    "confusion_matrix": test_cm.tolist(),
    "n_test": int(len(all_labels)),
    "n_test_R": int((all_labels == 1).sum()),
    "n_test_S": int((all_labels == 0).sum()),
}
with open(os.path.join(MODEL_DIR, "test_results.json"), "w") as f:
    json.dump(test_results, f, indent=2)

# Save test predictions for explainability pipeline
np.savez(
    os.path.join(MODEL_DIR, "test_outputs.npz"),
    preds=all_preds,
    probs=all_probs,
    labels=all_labels,
    attn_weights=all_attn,
    gate_values=all_gate,
    test_ids=sid_test,
    gene_names=np.array(CATALOGUE_GENES),
    gene_presence=gp_test,
)
print(f"  Saved test outputs to model/test_outputs.npz")

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.1f}s")
print("DONE")
