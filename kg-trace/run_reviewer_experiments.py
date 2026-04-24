"""
Reviewer Experiments — All 6 experiments in one script.
Run from KG-Trace/ directory with the miniconda Python.

Exp 1: Auxiliary head ablation (full model, loss_gene_weight=0)
Exp 2: Random Forest with n_estimators=200
Exp 3: (Documentation only — handled in paper)
Exp 4: Ablations with max_epochs=100, patience=10
Exp 5: Baseline grid search
Exp 6: (Covered by Exp 4)
"""
import sys, os, time, json, csv, gc
sys.path.insert(0, os.path.dirname(__file__))
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM, PROJECT_DIR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, make_scorer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

assert torch.backends.mps.is_available(), "MPS not available"
DEVICE = "mps"

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
BASELINES_DIR = os.path.join(PROJECT_DIR, "baselines")
RESULTS_DIR = os.path.join(EVALUATE_DIR, "reviewer_experiments")
os.makedirs(RESULTS_DIR, exist_ok=True)

DRUG = "INH"

# ═══════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("LOADING SHARED DATA")
print("=" * 70)
t0_global = time.time()

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

KMER_DIM = X.shape[1]
print(f"  Dataset: {X.shape[0]} × {KMER_DIM}")

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

# Load EXACT same split
with open(os.path.join(MODEL_DIR, "split_ids.json")) as f:
    split_info = json.load(f)

sid_to_idx = {s: i for i, s in enumerate(labeled_samples)}
idx_train = np.array([sid_to_idx[s] for s in split_info["train_ids"] if s in sid_to_idx])
idx_val = np.array([sid_to_idx[s] for s in split_info["val_ids"] if s in sid_to_idx])
idx_test = np.array([sid_to_idx[s] for s in split_info["test_ids"] if s in sid_to_idx])

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
ge_train = gene_embeds_all[idx_train]
ge_val = gene_embeds_all[idx_val]
ge_test = gene_embeds_all[idx_test]
gp_train = gene_presence[idx_train]
gp_val = gene_presence[idx_val]
gp_test = gene_presence[idx_test]

print(f"  Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
print(f"  Data loading: {time.time()-t0_global:.1f}s\n")

# ── PyTorch Dataset ──────────────────────────────────────────────────────────
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

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)


def evaluate_model(model, test_loader):
    """Evaluate a PL model on the test loader, return AUROC and F1-macro."""
    model.eval()
    model = model.to("cpu")
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            kmer_x, gene_embeds, y_amr, _ = batch
            logits, _, _, _ = model(kmer_x, gene_embeds)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_probs.append(probs[:, 1].numpy())
            all_preds.append(preds.numpy())
            all_labels.append(y_amr.numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_probs)
    f1_mac = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds).tolist()
    return auroc, f1_mac, cm


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Auxiliary Head Ablation
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 1: Auxiliary Head Ablation (loss_gene_weight=0)")
print("=" * 70)
t1 = time.time()


def f1_score_torch(preds, targets, num_classes=2):
    return torch.tensor(f1_score(targets.cpu(), preds.cpu(), average="macro"), device=preds.device)


class KGTraceNoAux(pl.LightningModule):
    """Full KG-Trace architecture but with gene_head loss weight = 0."""
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.gene_attn_q = nn.Linear(256, KG_EMBED_DIM, bias=False)
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
        query = self.gene_attn_q(g).unsqueeze(2)
        attn_scores = torch.bmm(gene_embeds, query) / (gene_embeds.shape[-1] ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=1)
        k = (attn_weights * gene_embeds).sum(dim=1)
        g_p = self.proj_g(g)
        k_p = self.proj_k(k)
        gate = self.gate_mlp(torch.cat([g_p, k_p], dim=-1))
        fused = gate * g_p + (1 - gate) * k_p
        logits = self.classifier(fused)
        gene_preds = self.gene_head(fused)
        return logits, gene_preds, attn_weights.squeeze(-1), gate

    def training_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, y_genes = batch
        logits, gene_preds, _, _ = self(kmer_x, gene_embeds)
        loss_amr = nn.CrossEntropyLoss()(logits, y_amr)
        # *** KEY CHANGE: No auxiliary gene-head loss ***
        loss = loss_amr
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


model_noaux = KGTraceNoAux(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
ckpt_noaux = os.path.join(CKPT_DIR, "exp1_noaux")
os.makedirs(ckpt_noaux, exist_ok=True)

trainer_noaux = pl.Trainer(
    max_epochs=10,
    accelerator="mps", devices=1,
    callbacks=[ModelCheckpoint(dirpath=ckpt_noaux, filename="best", monitor="val_f1_macro", mode="max", save_top_k=1)],
    enable_progress_bar=True, enable_model_summary=False, logger=False,
)
trainer_noaux.fit(model_noaux, train_loader, val_loader)

best_ckpt = trainer_noaux.checkpoint_callback.best_model_path
if best_ckpt:
    model_noaux = KGTraceNoAux.load_from_checkpoint(best_ckpt, kmer_dim=KMER_DIM, num_genes=NUM_GENES)

auroc_noaux, f1_noaux, cm_noaux = evaluate_model(model_noaux, test_loader)

exp1_result = {
    "experiment": "Exp1_NoAuxHead",
    "description": "Full KG-Trace, loss = loss_amr only (no 0.3*loss_gene)",
    "auroc": float(auroc_noaux),
    "f1_macro": float(f1_noaux),
    "confusion_matrix": cm_noaux,
    "train_time_s": round(time.time() - t1, 1),
}
print(f"\n  Exp 1 Result: AUROC={auroc_noaux:.4f}, F1-macro={f1_noaux:.4f}")
print(f"  Time: {exp1_result['train_time_s']}s")

del model_noaux, trainer_noaux
gc.collect()
torch.mps.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Random Forest with n_estimators=200
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 2: Random Forest with n_estimators=200")
print("=" * 70)
t2 = time.time()

# Baselines use train+val
train_val_ids = set(split_info["train_ids"]) | set(split_info["val_ids"])
test_ids_set = set(split_info["test_ids"])
train_val_samples = [s for s in all_samples if s in train_val_ids]
test_samples = [s for s in all_samples if s in test_ids_set]
tv_indices = [sample_to_idx[s] for s in train_val_samples]
te_indices = [sample_to_idx[s] for s in test_samples]

X_train_bl = X_sparse[tv_indices].toarray().astype(np.float32)
X_test_bl = X_sparse[te_indices].toarray().astype(np.float32)
y_train_bl = labels_df.loc[train_val_samples, "label"].values.astype(int)
y_test_bl = labels_df.loc[test_samples, "label"].values.astype(int)

print(f"  Train+Val: {X_train_bl.shape[0]}, Test: {X_test_bl.shape[0]}")

rf200 = RandomForestClassifier(
    n_estimators=200, max_depth=20, max_features="sqrt",
    random_state=42, n_jobs=-1
)
rf200.fit(X_train_bl, y_train_bl)
rf200_preds = rf200.predict(X_test_bl)
rf200_probs = rf200.predict_proba(X_test_bl)[:, 1]

exp2_result = {
    "experiment": "Exp2_RF200",
    "description": "Random Forest with n_estimators=200 (matching paper claim)",
    "auroc": float(roc_auc_score(y_test_bl, rf200_probs)),
    "f1_macro": float(f1_score(y_test_bl, rf200_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test_bl, rf200_preds).tolist(),
    "train_time_s": round(time.time() - t2, 1),
}
print(f"\n  Exp 2 Result: AUROC={exp2_result['auroc']:.4f}, F1-macro={exp2_result['f1_macro']:.4f}")
print(f"  Time: {exp2_result['train_time_s']}s")

del rf200
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Ablations with max_epochs=100, patience=10
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 4: Ablations with max_epochs=100, patience=10")
print("=" * 70)

# Import the original model
from model.kg_trace import KGTrace


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
        self.log("train_loss", loss); return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))
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
        self.log("train_loss", loss); return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))
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
        k = gene_embeds.mean(dim=1)
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
        self.log("train_loss", loss); return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ScalarFusion(pl.LightningModule):
    """Scalar α·g + (1-α)·k instead of cross-attention gate."""
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
        self.alpha = nn.Parameter(torch.tensor(0.5))
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
        self.log("train_loss", loss); return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


ABLATION_CONFIGS = [
    ("genomic_only", "k-mer features only, no KG branch", GenomicOnly),
    ("kg_only", "KG embeddings only, no k-mer branch", KGOnly),
    ("avg_pool_no_attention", "Average pooling instead of self-attention", AvgPoolFusion),
    ("scalar_fusion", "Scalar α·g + (1-α)·k instead of cross-attention gate", ScalarFusion),
    ("full_kg_trace", "Full KG-Trace (100 epochs, early stopping)", KGTrace),
]

ablation_results_100ep = []

for config_name, description, ModelClass in ABLATION_CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Ablation: {config_name} — {description}")
    print(f"{'='*60}")
    t_start = time.time()

    model = ModelClass(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    ckpt_path = os.path.join(CKPT_DIR, f"exp4_{config_name}")
    os.makedirs(ckpt_path, exist_ok=True)

    early_stop = EarlyStopping(monitor="val_f1_macro", mode="max", patience=10, verbose=False)
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_path, filename="best", monitor="val_f1_macro",
        mode="max", save_top_k=1, verbose=False,
    )

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="mps", devices=1,
        callbacks=[early_stop, checkpoint_cb],
        enable_progress_bar=True, enable_model_summary=False, logger=False,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    stopped_epoch = trainer.current_epoch
    best_score = checkpoint_cb.best_model_score

    if best_ckpt:
        model = ModelClass.load_from_checkpoint(best_ckpt, kmer_dim=KMER_DIM, num_genes=NUM_GENES)

    auroc, f1_mac, cm = evaluate_model(model, test_loader)
    elapsed = time.time() - t_start

    result = {
        "config": config_name,
        "description": description,
        "auroc": float(auroc),
        "f1_macro": float(f1_mac),
        "confusion_matrix": cm,
        "n_params": total_params,
        "stopped_epoch": stopped_epoch,
        "best_val_f1": float(best_score) if best_score else None,
        "train_time_s": round(elapsed, 1),
    }
    ablation_results_100ep.append(result)

    print(f"  AUROC: {auroc:.4f}, F1: {f1_mac:.4f}")
    print(f"  Stopped at epoch {stopped_epoch}, best val_f1={best_score:.4f}" if best_score else "")
    print(f"  Time: {elapsed:.1f}s")

    del model, trainer
    gc.collect()
    torch.mps.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Baseline Grid Search
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EXPERIMENT 5: Baseline Grid Search (tuned on validation set)")
print("=" * 70)
t5 = time.time()

# For grid search, use train-only for fitting, val for selection, test for eval
# Need to separate train from val for baselines
X_train_only = X_sparse[[sample_to_idx[s] for s in split_info["train_ids"] if s in sample_to_idx]].toarray().astype(np.float32)
y_train_only = labels_df.loc[[s for s in split_info["train_ids"] if s in set(labels_df.index)], "label"].values.astype(int)
X_val_only = X_sparse[[sample_to_idx[s] for s in split_info["val_ids"] if s in sample_to_idx]].toarray().astype(np.float32)
y_val_only = labels_df.loc[[s for s in split_info["val_ids"] if s in set(labels_df.index)], "label"].values.astype(int)

print(f"  Train-only: {X_train_only.shape[0]}, Val: {X_val_only.shape[0]}, Test: {X_test_bl.shape[0]}")

# Combine train+val for final refit after selecting best hyperparams
X_trainval = np.vstack([X_train_only, X_val_only])
y_trainval = np.concatenate([y_train_only, y_val_only])

grid_results = []

# --- LinearSVC Grid ---
print("\n  --- LinearSVC Grid Search ---")
t_svm = time.time()
best_svm_f1 = -1
best_svm_C = None
for C in [0.01, 0.1, 1.0, 10.0]:
    svm_model = LinearSVC(C=C, max_iter=5000, random_state=42, dual="auto")
    svm_model.fit(X_train_only, y_train_only)
    val_preds = svm_model.predict(X_val_only)
    val_f1 = f1_score(y_val_only, val_preds, average="macro")
    print(f"    C={C}: val F1={val_f1:.4f}")
    if val_f1 > best_svm_f1:
        best_svm_f1 = val_f1
        best_svm_C = C

print(f"    Best C={best_svm_C} (val F1={best_svm_f1:.4f})")
# Retrain on train+val with best C, then evaluate on test
svm_final = LinearSVC(C=best_svm_C, max_iter=5000, random_state=42, dual="auto")
svm_cal = CalibratedClassifierCV(svm_final, cv=3)
svm_cal.fit(X_trainval, y_trainval)
svm_test_preds = svm_cal.predict(X_test_bl)
svm_test_probs = svm_cal.predict_proba(X_test_bl)[:, 1]

svm_grid_result = {
    "model": "SVM (tuned)",
    "best_C": best_svm_C,
    "auroc": float(roc_auc_score(y_test_bl, svm_test_probs)),
    "f1_macro": float(f1_score(y_test_bl, svm_test_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test_bl, svm_test_preds).tolist(),
    "search_space": {"C": [0.01, 0.1, 1.0, 10.0]},
    "train_time_s": round(time.time() - t_svm, 1),
}
grid_results.append(svm_grid_result)
print(f"    Test: AUROC={svm_grid_result['auroc']:.4f}, F1={svm_grid_result['f1_macro']:.4f}")

del svm_final, svm_cal
gc.collect()

# --- XGBoost Grid ---
print("\n  --- XGBoost Grid Search ---")
t_xgb = time.time()
best_xgb_f1 = -1
best_xgb_params = None
for n_est in [100, 200]:
    for depth in [4, 6, 8]:
        xgb_m = xgb.XGBClassifier(
            n_estimators=n_est, max_depth=depth, learning_rate=0.1,
            random_state=42, eval_metric="logloss", tree_method="hist"
        )
        xgb_m.fit(X_train_only, y_train_only)
        val_preds = xgb_m.predict(X_val_only)
        val_f1 = f1_score(y_val_only, val_preds, average="macro")
        print(f"    n_est={n_est}, depth={depth}: val F1={val_f1:.4f}")
        if val_f1 > best_xgb_f1:
            best_xgb_f1 = val_f1
            best_xgb_params = {"n_estimators": n_est, "max_depth": depth}
        del xgb_m
        gc.collect()

print(f"    Best: {best_xgb_params} (val F1={best_xgb_f1:.4f})")
xgb_final = xgb.XGBClassifier(
    n_estimators=best_xgb_params["n_estimators"],
    max_depth=best_xgb_params["max_depth"],
    learning_rate=0.1, random_state=42, eval_metric="logloss", tree_method="hist"
)
xgb_final.fit(X_trainval, y_trainval)
xgb_test_preds = xgb_final.predict(X_test_bl)
xgb_test_probs = xgb_final.predict_proba(X_test_bl)[:, 1]

xgb_grid_result = {
    "model": "XGBoost (tuned)",
    "best_params": best_xgb_params,
    "auroc": float(roc_auc_score(y_test_bl, xgb_test_probs)),
    "f1_macro": float(f1_score(y_test_bl, xgb_test_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test_bl, xgb_test_preds).tolist(),
    "search_space": {"n_estimators": [100, 200], "max_depth": [4, 6, 8]},
    "train_time_s": round(time.time() - t_xgb, 1),
}
grid_results.append(xgb_grid_result)
print(f"    Test: AUROC={xgb_grid_result['auroc']:.4f}, F1={xgb_grid_result['f1_macro']:.4f}")

del xgb_final
gc.collect()

# --- Random Forest Grid ---
print("\n  --- Random Forest Grid Search ---")
t_rf = time.time()
best_rf_f1 = -1
best_rf_params = None
for n_est in [100, 200]:
    for depth in [10, 20, None]:
        rf_m = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, max_features="sqrt",
            random_state=42, n_jobs=-1
        )
        rf_m.fit(X_train_only, y_train_only)
        val_preds = rf_m.predict(X_val_only)
        val_f1 = f1_score(y_val_only, val_preds, average="macro")
        depth_str = str(depth) if depth else "None"
        print(f"    n_est={n_est}, depth={depth_str}: val F1={val_f1:.4f}")
        if val_f1 > best_rf_f1:
            best_rf_f1 = val_f1
            best_rf_params = {"n_estimators": n_est, "max_depth": depth}
        del rf_m
        gc.collect()

print(f"    Best: {best_rf_params} (val F1={best_rf_f1:.4f})")
rf_final = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"],
    max_depth=best_rf_params["max_depth"],
    max_features="sqrt", random_state=42, n_jobs=-1
)
rf_final.fit(X_trainval, y_trainval)
rf_test_preds = rf_final.predict(X_test_bl)
rf_test_probs = rf_final.predict_proba(X_test_bl)[:, 1]

rf_grid_result = {
    "model": "RandomForest (tuned)",
    "best_params": best_rf_params,
    "auroc": float(roc_auc_score(y_test_bl, rf_test_probs)),
    "f1_macro": float(f1_score(y_test_bl, rf_test_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test_bl, rf_test_preds).tolist(),
    "search_space": {"n_estimators": [100, 200], "max_depth": [10, 20, "None"]},
    "train_time_s": round(time.time() - t_rf, 1),
}
grid_results.append(rf_grid_result)
print(f"    Test: AUROC={rf_grid_result['auroc']:.4f}, F1={rf_grid_result['f1_macro']:.4f}")

del rf_final
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════
# SAVE ALL RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SAVING ALL RESULTS")
print("=" * 70)

all_results = {
    "experiment_1_aux_head_ablation": exp1_result,
    "experiment_2_rf200": exp2_result,
    "experiment_3_note": "Documentation only: baselines trained on train+val (32,096 samples). KG-Trace uses train-only (26,432 samples). State this in paper.",
    "experiment_4_ablations_100ep": ablation_results_100ep,
    "experiment_5_grid_search": grid_results,
    "experiment_6_note": "Covered by experiment 4 (100 epochs with early stopping).",
    "total_time_s": round(time.time() - t0_global, 1),
}

results_path = os.path.join(RESULTS_DIR, "all_reviewer_experiments.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# ── Print summary table ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print("\n--- Exp 1: Aux Head Ablation ---")
print(f"  Full KG-Trace WITH aux head:    AUROC=0.9757, F1=0.9594  (from test_results.json)")
print(f"  Full KG-Trace WITHOUT aux head: AUROC={exp1_result['auroc']:.4f}, F1={exp1_result['f1_macro']:.4f}")

print("\n--- Exp 2: RF n_estimators=200 ---")
print(f"  RF (100 trees, old):  AUROC=0.9604, F1=0.8350  (from baseline_results.json)")
print(f"  RF (200 trees, new):  AUROC={exp2_result['auroc']:.4f}, F1={exp2_result['f1_macro']:.4f}")

print("\n--- Exp 4: Ablations (100 epochs, early stopping patience=10) ---")
print(f"  {'Config':<25s} {'AUROC':>8s} {'F1':>8s} {'Stopped':>8s}")
print(f"  {'-'*55}")
full_auroc_100 = None
for r in ablation_results_100ep:
    if r["config"] == "full_kg_trace":
        full_auroc_100 = r["auroc"]
for r in ablation_results_100ep:
    delta = r["auroc"] - full_auroc_100 if full_auroc_100 else 0.0
    print(f"  {r['config']:<25s} {r['auroc']:8.4f} {r['f1_macro']:8.4f} ep={r['stopped_epoch']:>3d}")

print("\n--- Exp 5: Baseline Grid Search ---")
print(f"  {'Model':<25s} {'AUROC':>8s} {'F1-macro':>10s} {'Best Params'}")
print(f"  {'-'*65}")
for r in grid_results:
    params = r.get("best_params", {"C": r.get("best_C")})
    print(f"  {r['model']:<25s} {r['auroc']:8.4f} {r['f1_macro']:10.4f} {params}")

print(f"\n  Total experiment time: {all_results['total_time_s']:.1f}s")
print(f"  Results saved to: {results_path}")
print("\nALL EXPERIMENTS DONE")
