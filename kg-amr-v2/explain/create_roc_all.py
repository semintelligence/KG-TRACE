#!/usr/bin/env python3
"""
Generate ROC curves with ALL baselines + ALL ablations.
Panel A: KG-TRACE vs Baselines (SVM, XGBoost, RF)
Panel B: KG-TRACE vs Ablations (genomic_only, kg_only, avg_pool, scalar_fusion, no_aux_head)
"""
import sys, os, json, gc
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # kg-amr-v2
sys.path.insert(0, PROJECT_DIR)
from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints")
FIG_DIR = os.path.join(PROJECT_DIR, "explain", "final_publication_figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD SAVED PREDICTIONS (baselines + KG-TRACE primary)
# ═══════════════════════════════════════════════════════════════════
print("[1/4] Loading saved predictions...")
to = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
bp = np.load(os.path.join(PROJECT_DIR, "baselines", "baseline_predictions.npz"), allow_pickle=True)

kgtrace_probs = to["probs"]
kgtrace_labels = to["labels"]
svm_probs = bp["svm_probs"]
xgb_probs = bp["xgb_probs"]
rf_probs = bp["rf_probs"]
bl_labels = bp["y_test"]

# ═══════════════════════════════════════════════════════════════════
# 2. RELOAD ABLATION CHECKPOINTS & RUN INFERENCE
# ═══════════════════════════════════════════════════════════════════
print("[2/4] Loading data for ablation inference...")

X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

labels_df = pd.read_parquet(os.path.join(FEATURES_DIR, "labels/labels_INH.parquet"))
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]

sample_to_idx = {s: i for i, s in enumerate(all_samples)}
labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
sample_indices = [sample_to_idx[s] for s in labeled_samples]
X = X_sparse[sample_indices].toarray().astype(np.float32)
y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)

KMER_DIM = X.shape[1]

# KG embeddings
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
entity_emb = np.abs(entity_emb_raw).astype(np.float32) if np.iscomplexobj(entity_emb_raw) else entity_emb_raw.astype(np.float32)

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

with open(os.path.join(MODEL_DIR, "split_ids.json")) as f:
    split_info = json.load(f)

sid_to_idx = {s: i for i, s in enumerate(labeled_samples)}
idx_test = np.array([sid_to_idx[s] for s in split_info["test_ids"] if s in sid_to_idx])
X_test = X[idx_test]
y_test = y[idx_test]
ge_test = gene_embeds_all[idx_test]
gp_test = gene_presence[idx_test]

class AMRDataset(Dataset):
    def __init__(self, X, gene_embeds, y_amr, gene_presence):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.gene_embeds = torch.tensor(gene_embeds, dtype=torch.float32)
        self.y_amr = torch.tensor(y_amr, dtype=torch.long)
        self.gene_presence = torch.tensor(gene_presence, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.gene_embeds[idx], self.y_amr[idx], self.gene_presence[idx]

test_ds = AMRDataset(X_test, ge_test, y_test, gp_test)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

# ── Define ablation model classes (same as run_reviewer_experiments.py) ──
import pytorch_lightning as pl
from sklearn.metrics import f1_score as sklearn_f1

def f1_score_torch(preds, targets, num_classes=2):
    return torch.tensor(sklearn_f1(targets.cpu(), preds.cpu(), average="macro"), device=preds.device)

class GenomicOnly(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid())
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
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

class KGOnly(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid())
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
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

class AvgPoolFusion(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256))
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.gate_mlp = nn.Sequential(nn.Linear(FUSED_DIM * 2, FUSED_DIM), nn.ReLU(), nn.Linear(FUSED_DIM, FUSED_DIM), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid())
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
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

class ScalarFusion(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256))
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid())
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
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

class KGAMRNoAux(pl.LightningModule):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.save_hyperparameters()
        self.genomic_encoder = nn.Sequential(nn.Linear(kmer_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 256))
        self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
        self.gene_attn_q = nn.Linear(256, KG_EMBED_DIM, bias=False)
        self.proj_g = nn.Linear(256, FUSED_DIM)
        self.proj_k = nn.Linear(KG_EMBED_DIM, FUSED_DIM)
        self.gate_mlp = nn.Sequential(nn.Linear(FUSED_DIM * 2, FUSED_DIM), nn.ReLU(), nn.Linear(FUSED_DIM, FUSED_DIM), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, 2))
        self.gene_head = nn.Sequential(nn.Linear(FUSED_DIM, 64), nn.ReLU(), nn.Linear(64, num_genes), nn.Sigmoid())
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
        loss = nn.CrossEntropyLoss()(logits, y_amr)
        self.log("train_loss", loss); return loss
    def validation_step(self, batch, _):
        kmer_x, gene_embeds, y_amr, _ = batch
        logits, _, _, _ = self(kmer_x, gene_embeds)
        preds = torch.argmax(logits, dim=1)
        self.log("val_f1_macro", f1_score_torch(preds, y_amr))
    def configure_optimizers(self): return torch.optim.Adam(self.parameters(), lr=1e-3)

# ── Run inference on each ablation checkpoint ──
def get_probs_from_checkpoint(ModelClass, ckpt_path):
    model = ModelClass.load_from_checkpoint(ckpt_path, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
    model.eval()
    model = model.to("cpu")
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            kmer_x, gene_embeds, y_amr, _ = batch
            logits, _, _, _ = model(kmer_x, gene_embeds)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs[:, 1].numpy())
    del model
    gc.collect()
    return np.concatenate(all_probs)

print("\n[3/4] Running inference on ablation checkpoints...")

# Use exp4 checkpoints (100 epoch, early stop — matching the paper table)
ablation_probs = {}
ABLATION_CKPTS = [
    ("genomic_only", GenomicOnly, os.path.join(CKPT_DIR, "exp4_genomic_only", "best.ckpt")),
    ("kg_only", KGOnly, os.path.join(CKPT_DIR, "exp4_kg_only", "best.ckpt")),
    ("avg_pool", AvgPoolFusion, os.path.join(CKPT_DIR, "exp4_avg_pool_no_attention", "best.ckpt")),
    ("scalar_fusion", ScalarFusion, os.path.join(CKPT_DIR, "exp4_scalar_fusion", "best.ckpt")),
    ("no_aux_head", KGAMRNoAux, os.path.join(CKPT_DIR, "exp1_noaux", "best.ckpt")),
]

for name, ModelClass, ckpt_path in ABLATION_CKPTS:
    if os.path.exists(ckpt_path):
        print(f"  Loading {name} from {ckpt_path}...")
        ablation_probs[name] = get_probs_from_checkpoint(ModelClass, ckpt_path)
        print(f"    ✓ {name}: {len(ablation_probs[name])} samples")
    else:
        print(f"  ✗ {name}: checkpoint not found at {ckpt_path}")

# ═══════════════════════════════════════════════════════════════════
# 3. PLOT TWO-PANEL ROC FIGURE
# ═══════════════════════════════════════════════════════════════════
print("\n[4/4] Creating two-panel ROC figure...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# ── Color palettes ──
baseline_colors = {
    "KG-TRACE (primary)": "#E74C3C",
    "LinearSVC (C=1.0)": "#2980B9",
    "XGBoost (n=200, d8)": "#27AE60",
    "Random Forest (n=200)": "#8E44AD",
}
ablation_colors = {
    "KG-TRACE (primary)": "#E74C3C",
    "genomic_only": "#3498DB",
    "kg_only": "#E67E22",
    "avg_pool": "#9B59B6",
    "scalar_fusion": "#1ABC9C",
    "no_aux_head": "#F39C12",
}
ablation_styles = {
    "KG-TRACE (primary)": "-",
    "genomic_only": "--",
    "kg_only": "-.",
    "avg_pool": "--",
    "scalar_fusion": "-.",
    "no_aux_head": ":",
}

# ── Panel A: Baselines ──
ax1.set_title("A) ROC Curves — Baselines", fontsize=16, fontweight="bold", pad=15)

baseline_data = [
    ("KG-TRACE (primary)", kgtrace_probs, kgtrace_labels),
    ("LinearSVC (C=1.0)", svm_probs, bl_labels),
    ("XGBoost (n=200, d8)", xgb_probs, bl_labels),
    ("Random Forest (n=200)", rf_probs, bl_labels),
]

for name, probs, labels in baseline_data:
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    lw = 3.0 if name == "KG-TRACE (primary)" else 2.0
    ax1.plot(fpr, tpr, color=baseline_colors[name], lw=lw, label=f"{name} (AUC = {roc_auc:.4f})")

ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1, alpha=0.7)
ax1.set_xlabel("False Positive Rate", fontsize=13)
ax1.set_ylabel("True Positive Rate", fontsize=13)
ax1.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax1.set_xlim([-0.01, 1.01])
ax1.set_ylim([-0.01, 1.01])
ax1.grid(True, alpha=0.3)

# ── Panel B: Ablations ──
ax2.set_title("B) ROC Curves — Ablation Study", fontsize=16, fontweight="bold", pad=15)

# Plot KG-TRACE primary first
fpr, tpr, _ = roc_curve(kgtrace_labels, kgtrace_probs)
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color=ablation_colors["KG-TRACE (primary)"], lw=3.0, label=f"KG-TRACE (primary) (AUC = {roc_auc:.4f})")

# Plot each ablation
ablation_labels_test = y_test  # ablations use the same test set
for name in ["genomic_only", "kg_only", "avg_pool", "scalar_fusion", "no_aux_head"]:
    if name in ablation_probs:
        fpr, tpr, _ = roc_curve(ablation_labels_test, ablation_probs[name])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=ablation_colors[name], lw=2.0,
                 linestyle=ablation_styles[name],
                 label=f"{name} (AUC = {roc_auc:.4f})")

ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1, alpha=0.7)
ax2.set_xlabel("False Positive Rate", fontsize=13)
ax2.set_ylabel("True Positive Rate", fontsize=13)
ax2.legend(loc="lower right", fontsize=10, framealpha=0.9)
ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.01])
ax2.grid(True, alpha=0.3)

plt.suptitle("Figure 3: Test Set ROC Curves (n = 5,665)", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()

# Save
out_png = os.path.join(FIG_DIR, "Fig3_ROC_All.png")
out_pdf = os.path.join(FIG_DIR, "Fig3_ROC_All.pdf")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"\n✓ Saved: {out_png}")
print(f"✓ Saved: {out_pdf}")
print("DONE")
