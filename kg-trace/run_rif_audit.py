import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from paths import KG_EMBED_DIM, FUSED_DIM, PROJECT_DIR

import numpy as np
import torch
import networkx as nx
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from model.kg_amr import KGTrace

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")

DRUG = "RIF"
CKPT = os.path.join(MODEL_DIR, "checkpoints/best_model-v4.ckpt")

# 1. Load data
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

# 2. Gene embeddings
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
entity_emb = np.abs(entity_emb_raw).astype(np.float32) if np.iscomplexobj(entity_emb_raw) else entity_emb_raw.astype(np.float32)
with open(os.path.join(KG_DIR, "embeddings/pykeen_entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(KG_DIR, "gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
CATALOGUE_GENES = sorted(gene_mechanism.keys())
NUM_GENES = len(CATALOGUE_GENES)
gene_emb_indices = [entity_to_id[g] for g in CATALOGUE_GENES]
gene_emb_matrix = entity_emb[gene_emb_indices]

gene_presence = np.zeros((X.shape[0], NUM_GENES), dtype=np.float32)
for g_idx, g_name in enumerate(CATALOGUE_GENES):
    prefix = g_name + ":"
    for f_idx, f_name in enumerate(all_features):
        if f_name.startswith(prefix):
            gene_presence[:, g_idx] = np.maximum(gene_presence[:, g_idx], X[:, f_idx])

# 3. Split
idx_train, idx_temp = train_test_split(np.arange(len(y)), test_size=0.30, random_state=42, stratify=y)
idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42, stratify=y[idx_temp])
X_test = X[idx_test]
y_test = y[idx_test]
gp_test = gene_presence[idx_test]

# 4. Model
KMER_DIM = len(all_features)
model = KGTrace.load_from_checkpoint(CKPT, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
model.eval()
model.to('cpu')

gene_embeds_test = np.zeros((len(idx_test), NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(len(idx_test)):
    for j in range(NUM_GENES):
        if gp_test[i, j] > 0:
            gene_embeds_test[i, j, :] = gene_emb_matrix[j]

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
ge_test_tensor = torch.tensor(gene_embeds_test, dtype=torch.float32)
shap_values = np.zeros_like(X_test)
all_preds = []

with torch.enable_grad():
    for i in range(len(idx_test)):
        x_i = X_test_tensor[i:i+1].clone().detach().requires_grad_(True)
        ge_i = ge_test_tensor[i:i+1]
        logits, _, _, _ = model(x_i, ge_i)
        all_preds.append(torch.argmax(logits, dim=1).item())
        logits[0, 1].backward()
        shap_values[i] = x_i.grad.detach().numpy()[0] * X_test[i]
all_preds = np.array(all_preds)

# 5. Graph
G = nx.read_graphml(os.path.join(PROJECT_DIR, "kg/amr_graph.graphml"))
mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]

cache = {}
def has_kg_path(feat_name):
    if feat_name in cache: return cache[feat_name]
    result = 0
    if G.has_node(feat_name) and G.has_node(DRUG) and nx.has_path(G, feat_name, DRUG): result = 1
    else:
        gene = feat_name.split(":")[0]
        if G.has_node(gene) and G.has_node(DRUG) and nx.has_path(G, gene, DRUG): result = 1
    cache[feat_name] = result
    return result

# BGR@50
top_50 = [all_features[i] for i in sorted_indices[:50]]
grounded_50 = sum(has_kg_path(f) for f in top_50)
bgr50 = grounded_50 / 50.0

# Coverage
n_pred_r = (all_preds == 1).sum()
grounded_top1 = 0
for i in range(len(idx_test)):
    if all_preds[i] != 1: continue
    top_idx = np.argmax(np.abs(shap_values[i]))
    if has_kg_path(all_features[top_idx]): grounded_top1 += 1
coverage = grounded_top1 / n_pred_r if n_pred_r > 0 else 0

print(json.dumps({"drug": DRUG, "bgr50": bgr50, "coverage": coverage, "n_pred_r": int(n_pred_r), "grounded_top1": grounded_top1}))
