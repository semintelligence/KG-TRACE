import sys, os, time, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, PROJECT_DIR

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")

# Load data similarly
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

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

sample_to_idx = {s: i for i, s in enumerate(all_samples)}

# Load INH labels
label_path = os.path.join(FEATURES_DIR, "labels/labels_INH.parquet")
labels_df = pd.read_parquet(label_path)
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]

labeled_samples = [s for s in all_samples if s in set(labels_df.index)]
sample_indices = [sample_to_idx[s] for s in labeled_samples]
X = X_sparse[sample_indices].toarray().astype(np.float32)
y = labels_df.loc[labeled_samples, "label"].values.astype(np.int64)

# Gene presence
gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for fi, gi in feature_to_gene_idx.items():
    gene_presence[:, gi] = np.maximum(gene_presence[:, gi], X[:, fi])

gene_embeds_all = np.zeros((X.shape[0], NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(X.shape[0]):
    for j in range(NUM_GENES):
        if gene_presence[i, j] > 0:
            gene_embeds_all[i, j, :] = gene_emb_matrix[j]

indices = np.arange(X.shape[0])
idx_train, idx_temp = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])

# Create fully connected edge index for 26 nodes
edges = []
for i in range(NUM_GENES):
    for j in range(NUM_GENES):
        edges.append((i, j))
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

def create_pyg_dataset(indices):
    dataset = []
    for i in indices:
        x_nodes = torch.tensor(gene_embeds_all[i])
        y_val = torch.tensor([y[i]], dtype=torch.long)
        kmer_val = torch.tensor(X[i]).unsqueeze(0)
        data = Data(x=x_nodes, edge_index=edge_index, y=y_val, kmer=kmer_val)
        dataset.append(data)
    return dataset

print("Creating PyG Datasets...")
train_ds = create_pyg_dataset(idx_train)
val_ds = create_pyg_dataset(idx_val)
test_ds = create_pyg_dataset(idx_test)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

class PyGGATBaseline(torch.nn.Module):
    def __init__(self, kmer_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GATConv(64, 64, heads=4, concat=False)
        self.conv2 = GATConv(64, 64, heads=4, concat=False)
        self.kmer_mlp = nn.Sequential(
            nn.Linear(kmer_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph branch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # [B, 64]
        
        # Genomic branch
        kmer_feat = self.kmer_mlp(data.kmer) # [B, 128]
        
        # Concat
        fused = torch.cat([kmer_feat, x], dim=-1)
        return self.classifier(fused)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = PyGGATBaseline(kmer_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("Training GAT Baseline...")
best_val_f1 = 0
for epoch in range(15):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            preds = out.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(batch.y.cpu().numpy())
    
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), os.path.join(EVALUATE_DIR, "best_gat.pth"))
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

model.load_state_dict(torch.load(os.path.join(EVALUATE_DIR, "best_gat.pth")))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch)
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(batch.y.cpu().numpy())

auroc = roc_auc_score(all_labels, all_probs)
f1_mac = f1_score(all_labels, all_preds, average='macro')
print("=== PyG GAT Baseline Test ===")
print(f"AUROC: {auroc:.4f}")
print(f"F1-macro: {f1_mac:.4f}")
