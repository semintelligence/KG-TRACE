import sys, os, time, csv, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pytorch_lightning as pl

from paths import KG_EMBED_DIM, GENOMIC_HIDDEN, FUSED_DIM, PROJECT_DIR
from model.kg_amr import KGTrace

FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")

# Load Mutation matrix
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

# Labels for INH
DRUG = "INH"
labels_df = pd.read_parquet(os.path.join(FEATURES_DIR, f"labels/labels_{DRUG}.parquet"))
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
sample_indices = [sample_to_idx[s] for s in labeled_samples]
X = X_sparse[sample_indices].toarray().astype(np.float32)
y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)
sample_ids = np.array(labeled_samples)

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

gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for g_idx, g_name in enumerate(CATALOGUE_GENES):
    prefix = g_name + ":"
    for f_idx, f_name in enumerate(all_features):
        if f_name.startswith(prefix):
            gene_presence[:, g_idx] = np.maximum(gene_presence[:, g_idx], X[:, f_idx])

X_train, X_tmp, y_train, y_tmp, gp_train, gp_tmp, ids_train, ids_tmp = train_test_split(
    X, y, gene_presence, sample_ids, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test, gp_val, gp_test, ids_val, ids_test = train_test_split(
    X_tmp, y_tmp, gp_tmp, ids_tmp, test_size=0.5, random_state=42, stratify=y_tmp
)

class AMRDataset(Dataset):
    def __init__(self, X, gene_embeds, y_amr, gene_presence):
        self.X = torch.tensor(X)
        self.gene_embeds = torch.tensor(gene_embeds)
        self.y_amr = torch.tensor(y_amr, dtype=torch.long)
        self.gene_presence = torch.tensor(gene_presence)
    def __len__(self): return len(self.y_amr)
    def __getitem__(self, idx):
        return self.X[idx], self.gene_embeds, self.y_amr[idx], self.gene_presence[idx]

train_ds = AMRDataset(X_train, gene_emb_matrix, y_train, gp_train)
val_ds   = AMRDataset(X_val, gene_emb_matrix, y_val, gp_val)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

weights = [0.1, 0.3, 0.5, 1.0]
results = {}

class KGTraceGrid(KGTrace):
    def __init__(self, w, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_gene_weight = w
        
    def training_step(self, batch, batch_idx):
        kmer_feat, gene_embeds, label, gene_targets = batch
        logits, gene_logits, _, _ = self(kmer_feat, gene_embeds)
        loss_cls = nn.CrossEntropyLoss()(logits, label)
        loss_gene = nn.BCELoss()(gene_logits, gene_targets.float())
        loss = loss_cls + self.loss_gene_weight * loss_gene
        self.log('train_loss', loss, prog_bar=True)
        return loss

for w in weights:
    print(f"--- Running w={w} ---")
    model = KGTraceGrid(w=w, kmer_dim=X.shape[1], num_genes=NUM_GENES)
    trainer = pl.Trainer(max_epochs=5, accelerator="mps", devices=1, enable_progress_bar=False, logger=False)
    trainer.fit(model, train_loader, val_loader)
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            kmer_feat, gene_embeds, label, _ = batch
            kmer_feat = kmer_feat.to("mps")
            gene_embeds = gene_embeds.to("mps")
            model = model.to("mps")
            logits, _, _, _ = model(kmer_feat, gene_embeds)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"w={w} => F1={val_f1:.4f}")
    results[w] = val_f1

with open("grid_search_results.json", "w") as f:
    json.dump(results, f, indent=2)
