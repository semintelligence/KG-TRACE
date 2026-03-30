"""
Task 3: Ablation Study — 5 configurations, 10 epochs each.
Uses IDENTICAL train/test split (from split_ids.json).
All metrics computed from real model outputs.
"""
import sys, os, time, json, csv, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "kg-amr-v2" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

assert torch.backends.mps.is_available(), "MPS not available"
DEVICE = "mps"

FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
os.makedirs(EVALUATE_DIR, exist_ok=True)

DRUG = "INH"
MAX_EPOCHS = 10

# ── 1. Load data (same as train.py) ─────────────────────────────────────────
print("[1/4] Loading data...")
t0 = time.time()

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

print(f"  Dataset: {X.shape[0]} × {X.shape[1]}")

# KG embeddings
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

# Gene presence
feature_to_gene_idx = {}
for fi, feat in enumerate(all_features):
    gene_name = feat.split(":")[0]
    if gene_name in CATALOGUE_GENES:
        feature_to_gene_idx[fi] = CATALOGUE_GENES.index(gene_name)

gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for fi, gi in feature_to_gene_idx.items():
    gene_presence[:, gi] = np.maximum(gene_presence[:, gi], X[:, fi])

gene_embeds_all = np.zeros((X.shape[0], NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(X.shape[0]):
    for j in range(NUM_GENES):
        if gene_presence[i, j] > 0:
            gene_embeds_all[i, j, :] = gene_emb_matrix[j]

# Load EXACT same split as original training
with open(os.path.join(MODEL_DIR, "split_ids.json")) as f:
    split_info = json.load(f)

sid_to_idx = {s: i for i, s in enumerate(labeled_samples)}
idx_train = np.array([sid_to_idx[s] for s in split_info["train_ids"] if s in sid_to_idx])
idx_val = np.array([sid_to_idx[s] for s in split_info["val_ids"] if s in sid_to_idx])
idx_test = np.array([sid_to_idx[s] for s in split_info["test_ids"] if s in sid_to_idx])

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
ge_train, ge_val, ge_test = gene_embeds_all[idx_train], gene_embeds_all[idx_val], gene_embeds_all[idx_test]
gp_train, gp_val, gp_test = gene_presence[idx_train], gene_presence[idx_val], gene_presence[idx_test]

KMER_DIM = X.shape[1]
print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

# ── 2. Define ablated model variants ─────────────────────────────────────────
print("\n[2/4] Defining ablated model variants...")

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
    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss = nn.CrossEntropyLoss()(logits, y_amr) + 0.3 * nn.BCELoss()(gene_preds, y_genes.float())
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(y_amr.cpu(), preds.cpu(), average="macro")
        self.log("val_f1_macro", torch.tensor(f1, device=kmer_x.device))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class KGOnly(pl.LightningModule):
    """KG embeddings only, no k-mer branch."""
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid()
        )
    def forward(self, kmer_x, gene_embeds):
        attn_scores = self.gene_attn(gene_embeds)
        attn_weights = torch.softmax(attn_scores, dim=1)
        k = (attn_weights * gene_embeds).sum(dim=1)
        k_p = self.proj_k(k)
        logits = self.classifier(k_p)
        gene_preds = self.gene_head(k_p)
        dummy_gate = torch.full((kmer_x.shape[0], FUSED_DIM), 0.0, device=kmer_x.device)
        return logits, gene_preds, attn_weights.squeeze(-1), dummy_gate
    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss = nn.CrossEntropyLoss()(logits, y_amr) + 0.3 * nn.BCELoss()(gene_preds, y_genes.float())
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(y_amr.cpu(), preds.cpu(), average="macro")
        self.log("val_f1_macro", torch.tensor(f1, device=kmer_x.device))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class AvgPoolFusion(pl.LightningModule):
    """Average pooling instead of self-attention for KG branch."""
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.gate_mlp = nn.Sequential(
            nn.Linear(FUSED_DIM * 2, FUSED_DIM), nn.ReLU(),
            nn.Linear(FUSED_DIM, FUSED_DIM), nn.Sigmoid()
        )
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid()
        )
    def forward(self, kmer_x, gene_embeds):
        g = self.genomic_encoder(kmer_x)
        k = gene_embeds.mean(dim=1)  # AVG pool instead of attention
        g_p = self.proj_g(g)
        k_p = self.proj_k(k)
        gate = self.gate_mlp(torch.cat([g_p, k_p], dim=-1))
        fused = gate * g_p + (1 - gate) * k_p
        logits = self.classifier(fused)
        gene_preds = self.gene_head(fused)
        dummy_attn = torch.ones(kmer_x.shape[0], gene_embeds.shape[1], device=kmer_x.device) / gene_embeds.shape[1]
        return logits, gene_preds, dummy_attn, gate
    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss = nn.CrossEntropyLoss()(logits, y_amr) + 0.3 * nn.BCELoss()(gene_preds, y_genes.float())
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(y_amr.cpu(), preds.cpu(), average="macro")
        self.log("val_f1_macro", torch.tensor(f1, device=kmer_x.device))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class ScalarFusion(pl.LightningModule):
    """Scalar alpha1*g + alpha2*k instead of cross-attention gate."""
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
        )
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # learnable scalar gate
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(
            nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid()
        )
    def forward(self, kmer_x, gene_embeds):
        g = self.genomic_encoder(kmer_x)
        attn_scores = self.gene_attn(gene_embeds)
        attn_weights = torch.softmax(attn_scores, dim=1)
        k = (attn_weights * gene_embeds).sum(dim=1)
        g_p = self.proj_g(g)
        k_p = self.proj_k(k)
        alpha = torch.sigmoid(self.alpha)
        fused = alpha * g_p + (1 - alpha) * k_p
        logits = self.classifier(fused)
        gene_preds = self.gene_head(fused)
        gate = torch.full((kmer_x.shape[0], FUSED_DIM), alpha.item(), device=kmer_x.device)
        return logits, gene_preds, attn_weights.squeeze(-1), gate
    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss = nn.CrossEntropyLoss()(logits, y_amr) + 0.3 * nn.BCELoss()(gene_preds, y_genes.float())
        self.log("train_loss", loss)
        return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        f1 = f1_score(y_amr.cpu(), preds.cpu(), average="macro")
        self.log("val_f1_macro", torch.tensor(f1, device=kmer_x.device))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Full KG-AMR v2 (re-import)
from model.kg_amr_v2 import KGAMRv2

ABLATION_CONFIGS = [
    ("genomic_only", "k-mer features only, no KG branch", GenomicOnly),
    ("kg_only", "KG embeddings only, no k-mer branch", KGOnly),
    ("avg_pool_no_attention", "Average pooling instead of self-attention", AvgPoolFusion),
    ("scalar_fusion", "Scalar α·g + (1-α)·k instead of cross-attention gate", ScalarFusion),
    ("full_kg_amr_v2", "Full KG-AMR v2 (10 epochs for comparison)", KGAMRv2),
]

# ── 3. Train all configurations ─────────────────────────────────────────────
print(f"\n[3/4] Training {len(ABLATION_CONFIGS)} ablation configurations ({MAX_EPOCHS} epochs each)...")

train_ds = AMRDataset(X_train, ge_train, y_train, gp_train)
val_ds = AMRDataset(X_val, ge_val, y_val, gp_val)
test_ds = AMRDataset(X_test, ge_test, y_test, gp_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

ablation_results = []

for config_name, description, ModelClass in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Config: {config_name} — {description}")
    print(f"{'='*60}")
    
    t_start = time.time()
    
    # Build model
    model = ModelClass(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Checkpoint
    ckpt_path = os.path.join(CKPT_DIR, f"ablation_{config_name}")
    os.makedirs(ckpt_path, exist_ok=True)
    
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path, filename="best", monitor="val_f1_macro",
        mode="max", save_top_k=1, verbose=False,
    )
    
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
    
    # Load best checkpoint
    best_ckpt = checkpoint_cb.best_model_path
    if best_ckpt:
        model = ModelClass.load_from_checkpoint(best_ckpt, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    model.eval()
    model = model.to("cpu")
    
    # Evaluate on test set
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
    
    auroc = roc_auc_score(all_labels, all_logits)
    f1_mac = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    elapsed = time.time() - t_start
    
    result = {
        "config": config_name,
        "description": description,
        "auroc": float(auroc),
        "f1_macro": float(f1_mac),
        "confusion_matrix": cm,
        "n_params": total_params,
        "train_time_s": round(elapsed, 1),
    }
    ablation_results.append(result)
    
    print(f"  AUROC: {auroc:.4f}")
    print(f"  F1-macro: {f1_mac:.4f}")
    print(f"  CM: {cm}")
    print(f"  Time: {elapsed:.1f}s")
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.mps.empty_cache()

# ── 4. Save results ─────────────────────────────────────────────────────────
print(f"\n[4/4] Saving ablation results...")

# Get full model AUROC for delta computation
full_auroc = None
for r in ablation_results:
    if r["config"] == "full_kg_amr_v2":
        full_auroc = r["auroc"]
        break

# CSV
csv_path = os.path.join(EVALUATE_DIR, "ablation_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Config", "Description", "AUROC", "F1-macro", "Delta_AUROC_vs_full", "N_params", "Train_time_s"])
    for r in ablation_results:
        delta = r["auroc"] - full_auroc if full_auroc else 0.0
        writer.writerow([
            r["config"], r["description"],
            f"{r['auroc']:.4f}", f"{r['f1_macro']:.4f}",
            f"{delta:+.4f}", r["n_params"], r["train_time_s"],
        ])

# JSON
json_path = os.path.join(EVALUATE_DIR, "ablation_results.json")
with open(json_path, "w") as f:
    json.dump(ablation_results, f, indent=2)

print(f"\n{'Config':<25s} {'AUROC':>8s} {'F1':>8s} {'Delta':>8s}")
print("-" * 55)
for r in ablation_results:
    delta = r["auroc"] - full_auroc if full_auroc else 0.0
    print(f"{r['config']:<25s} {r['auroc']:8.4f} {r['f1_macro']:8.4f} {delta:+8.4f}")

print(f"\n  Saved to: {csv_path}")
print(f"  Saved to: {json_path}")
print("DONE — run_ablation.py")
