"""
Regenerate INH test_outputs.npz from the original best_model.ckpt (Mar 27).
This script does NOT retrain — it only runs inference on the test split.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM, PROJECT_DIR

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report

from model.kg_amr import KGTrace

DEVICE = "mps"
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

DRUG = "INH"  # MUST be INH for the paper

# ── Load data (identical to train.py) ──
print("[1/4] Loading data...")
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

labels_df = pd.read_parquet(os.path.join(FEATURES_DIR, f"labels/labels_{DRUG}.parquet"))
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]

sample_to_idx = {s: i for i, s in enumerate(all_samples)}
labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
sample_indices = [sample_to_idx[s] for s in labeled_samples]
X = X_sparse[sample_indices].toarray().astype(np.float32)
y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)
sample_ids = np.array(labeled_samples)

print(f"  Dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Labels: {(y==1).sum()} R / {(y==0).sum()} S")

# ── Gene embeddings ──
print("[2/4] Building gene embeddings...")
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
if np.iscomplexobj(entity_emb_raw):
    entity_emb = np.abs(entity_emb_raw).astype(np.float32)
else:
    entity_emb = entity_emb_raw.astype(np.float32)

with open(os.path.join(KG_DIR, "embeddings/pykeen_entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(KG_DIR, "gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)

CATALOGUE_GENES = sorted(gene_mechanism.keys())
NUM_GENES = len(CATALOGUE_GENES)
gene_emb_indices = [entity_to_id.get(g) for g in CATALOGUE_GENES]
gene_emb_matrix = entity_emb[gene_emb_indices].astype(np.float32)

# Gene presence
gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for g_idx, g_name in enumerate(CATALOGUE_GENES):
    prefix = g_name + ":"
    for f_idx, f_name in enumerate(all_features):
        if f_name.startswith(prefix):
            gene_presence[:, g_idx] = np.maximum(gene_presence[:, g_idx], X[:, f_idx])

# ── Splits (same random_state=42) ──
print("[3/4] Splitting data...")
indices = np.arange(len(y))
idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])

X_test = X[idx_test]
y_test = y[idx_test]
gp_test = gene_presence[idx_test]
sid_test = sample_ids[idx_test]

print(f"  Test: {len(idx_test)} samples ({(y_test==1).sum()} R / {(y_test==0).sum()} S)")

class AMRDataset(Dataset):
    def __init__(self, X, gene_embeds, y_amr, gene_presence):
        self.X = torch.tensor(X)
        self.gene_embeds = torch.tensor(gene_embeds)
        self.y_amr = torch.tensor(y_amr, dtype=torch.long)
        self.gene_presence = torch.tensor(gene_presence)
    def __len__(self): return len(self.y_amr)
    def __getitem__(self, idx):
        return self.X[idx], self.gene_embeds, self.y_amr[idx], self.gene_presence[idx]

test_ds = AMRDataset(X_test, gene_emb_matrix, y_test, gp_test)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

# ── Load ORIGINAL INH checkpoint ──
print("[4/4] Loading original INH checkpoint and running inference...")
CKPT_PATH = os.path.join(MODEL_DIR, "checkpoints/best_model-v1.ckpt")
print(f"  Checkpoint: {CKPT_PATH}")

model = KGTrace.load_from_checkpoint(CKPT_PATH, kmer_dim=X.shape[1], num_genes=NUM_GENES)
model.eval()
model = model.to(DEVICE)

all_preds, all_probs, all_labels, all_attn, all_gate = [], [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        kmer_x, gene_embeds, label, _ = batch
        kmer_x = kmer_x.to(DEVICE)
        gene_embeds = gene_embeds.to(DEVICE)
        logits, _, attn, gate = model(kmer_x, gene_embeds)
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = torch.argmax(logits, dim=1)
        all_preds.append(pred.cpu().numpy())
        all_probs.append(prob.cpu().numpy())
        all_labels.append(label.numpy())
        all_attn.append(attn.cpu().numpy())
        all_gate.append(gate.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)
all_labels = np.concatenate(all_labels)
all_attn = np.concatenate(all_attn)
all_gate = np.concatenate(all_gate)

# Metrics
test_auroc = roc_auc_score(all_labels, all_probs)
test_f1 = f1_score(all_labels, all_preds, average="macro")
test_cm = confusion_matrix(all_labels, all_preds)

print(f"\n  === INH Test Results ===")
print(f"  AUROC:    {test_auroc:.4f}")
print(f"  F1-macro: {test_f1:.4f}")
print(f"  CM: {test_cm.tolist()}")
print(classification_report(all_labels, all_preds, target_names=['S','R']))

# Save
np.savez(
    os.path.join(MODEL_DIR, "test_outputs.npz"),
    preds=all_preds, probs=all_probs, labels=all_labels,
    attn_weights=all_attn, gate_values=all_gate,
    test_ids=sid_test, gene_names=np.array(CATALOGUE_GENES),
    gene_presence=gp_test,
)
print(f"  Saved INH test_outputs.npz")

# Gate analysis
gate_means = all_gate.mean(axis=1)
r_gates = gate_means[all_labels == 1]
s_gates = gate_means[all_labels == 0]
print(f"\n  Gate stats:")
print(f"  Resistant:   mean={r_gates.mean():.4f}, std={r_gates.std():.4f}")
print(f"  Susceptible: mean={s_gates.mean():.4f}, std={s_gates.std():.4f}")
print(f"  Overall:     mean={gate_means.mean():.4f}, std={gate_means.std():.4f}")
print("DONE")
