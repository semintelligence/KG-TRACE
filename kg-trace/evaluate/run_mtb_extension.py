"""
Task 4 Extension 2 — Multi-drug MTB extension.
Trains KG-Trace on additional CRyPTIC drugs using the SAME mutation matrix
and RotatE embeddings. Each drug gets a freshly initialized model for 10 epochs.
"""
import sys, os, time, json, csv, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, FUSED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

assert torch.backends.mps.is_available(), "MPS not available"
DEVICE = "mps"

FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
os.makedirs(EVALUATE_DIR, exist_ok=True)

MAX_EPOCHS = 10

# Drugs to evaluate: RIF is the primary target per user spec.
# Additional drugs included for multi-dataset comparison.
TARGET_DRUGS = ["RIF", "EMB", "LEV"]

# ── 1. Load SHARED data (mutation matrix + KG embeddings) ───────────────────
print("[1/4] Loading shared data...")
t0 = time.time()

X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

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
gene_emb_indices = [entity_to_id[g] for g in CATALOGUE_GENES]
gene_emb_matrix = entity_emb[gene_emb_indices].astype(np.float32)

feature_to_gene_idx = {}
for fi, feat in enumerate(all_features):
    gene_name = feat.split(":")[0]
    if gene_name in CATALOGUE_GENES:
        feature_to_gene_idx[fi] = CATALOGUE_GENES.index(gene_name)

sample_to_idx = {s: i for i, s in enumerate(all_samples)}
print(f"  Mutation matrix: {X_sparse.shape}, Genes: {NUM_GENES}, Embeddings: {entity_emb.shape}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── 2. Dataset class ────────────────────────────────────────────────────────

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

# ── 3. Train each drug ──────────────────────────────────────────────────────
print(f"\n[2/4] Training KG-Trace on {len(TARGET_DRUGS)} drugs ({MAX_EPOCHS} epochs each)...")

from model.kg_trace import KGTrace

all_results = []

for drug in TARGET_DRUGS:
    print(f"\n{'='*60}")
    print(f"  Drug: {drug}")
    print(f"{'='*60}")
    
    # Load labels
    label_path = os.path.join(FEATURES_DIR, f"labels/labels_{drug}.parquet")
    if not os.path.exists(label_path):
        print(f"  SKIP: {label_path} not found")
        continue
    
    labels_df = pd.read_parquet(label_path)
    labels_df = labels_df[~labels_df.index.duplicated(keep="first")]
    
    # Align with mutation matrix
    labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
    if len(labeled_samples) < 100:
        print(f"  SKIP: only {len(labeled_samples)} overlapping samples")
        continue
    
    sample_indices = [sample_to_idx[s] for s in labeled_samples]
    X = X_sparse[sample_indices].toarray().astype(np.float32)
    y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)
    
    n_r = int((y == 1).sum())
    n_s = int((y == 0).sum())
    print(f"  Samples: {len(y)} ({n_r} R / {n_s} S, {100*n_r/len(y):.1f}% R)")
    
    # Gene presence & embeddings
    gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
    for fi, gi in feature_to_gene_idx.items():
        gene_presence[:, gi] = np.maximum(gene_presence[:, gi], X[:, fi])
    
    gene_embeds_all = np.zeros((X.shape[0], NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
    for i in range(X.shape[0]):
        for j in range(NUM_GENES):
            if gene_presence[i, j] > 0:
                gene_embeds_all[i, j, :] = gene_emb_matrix[j]
    
    # Stratified split
    indices = np.arange(X.shape[0])
    idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])
    
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
    ge_train = gene_embeds_all[idx_train]
    ge_val = gene_embeds_all[idx_val]
    ge_test = gene_embeds_all[idx_test]
    gp_train = gene_presence[idx_train]
    gp_val = gene_presence[idx_val]
    gp_test = gene_presence[idx_test]
    
    print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
    
    train_ds = AMRDataset(X_train, ge_train, y_train, gp_train)
    val_ds = AMRDataset(X_val, ge_val, y_val, gp_val)
    test_ds = AMRDataset(X_test, ge_test, y_test, gp_test)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    
    KMER_DIM = X.shape[1]
    
    # Train
    model = KGTrace(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    
    ckpt_path = os.path.join(CKPT_DIR, f"MTB_{drug}")
    os.makedirs(ckpt_path, exist_ok=True)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path, filename="best_model",
        monitor="val_f1_macro", mode="max", save_top_k=1, verbose=False,
    )
    
    t_start = time.time()
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="mps",
        devices=1,
        callbacks=[checkpoint_cb],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    
    trainer.fit(model, train_loader, val_loader)
    train_time = time.time() - t_start
    
    # Load best checkpoint
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        model = KGTrace.load_from_checkpoint(best_ckpt, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    model.eval()
    model = model.to("cpu")
    
    # Evaluate
    all_logits = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            kmer_x, gene_embeds, y_amr, _ = batch
            logits, _, _, _ = model(kmer_x, gene_embeds)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_logits.append(probs[:, 1].numpy())
            all_preds.append(preds.numpy())
            all_labels.append(y_amr.numpy())
    
    all_logits = np.concatenate(all_logits)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    auroc = float(roc_auc_score(all_labels, all_logits))
    f1_mac = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    result = {
        "dataset": f"MTB_{drug}",
        "species": "M. tuberculosis",
        "drug": drug,
        "auroc": auroc,
        "f1_macro": f1_mac,
        "confusion_matrix": cm,
        "n_samples": len(y),
        "n_test": len(all_labels),
        "n_R": n_r,
        "n_S": n_s,
        "R_pct": round(100 * n_r / len(y), 1),
        "train_time_s": round(train_time, 1),
    }
    all_results.append(result)
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  F1-macro: {f1_mac:.4f}")
    print(f"  CM: {cm}")
    print(f"  Time: {train_time:.1f}s")
    print(f"  Checkpoint: {ckpt_path}")
    
    # Cleanup
    del model, trainer, X, y, gene_presence, gene_embeds_all
    del X_train, X_val, X_test, y_train, y_val, y_test
    del ge_train, ge_val, ge_test, gp_train, gp_val, gp_test
    del train_ds, val_ds, test_ds
    gc.collect()
    torch.mps.empty_cache()

# ── 4. Save results ─────────────────────────────────────────────────────────
print(f"\n[3/4] Saving MTB extension results...")

csv_path = os.path.join(EVALUATE_DIR, "mtb_extension_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Dataset", "Species", "Drug", "AUROC", "F1-macro", "N_samples", "N_test", "R_pct", "Train_time_s"])
    for r in all_results:
        writer.writerow([
            r["dataset"], r["species"], r["drug"],
            f"{r['auroc']:.4f}", f"{r['f1_macro']:.4f}",
            r["n_samples"], r["n_test"], r["R_pct"], r["train_time_s"],
        ])

json_path = os.path.join(EVALUATE_DIR, "mtb_extension_results.json")
with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'Dataset':<15s} {'AUROC':>8s} {'F1':>8s} {'N':>8s} {'R%':>6s}")
print("-" * 50)
for r in all_results:
    print(f"{r['dataset']:<15s} {r['auroc']:8.4f} {r['f1_macro']:8.4f} {r['n_samples']:8d} {r['R_pct']:5.1f}%")

print(f"\n  Saved to: {csv_path}")
print(f"  Saved to: {json_path}")
print("DONE — run_mtb_extension.py")
