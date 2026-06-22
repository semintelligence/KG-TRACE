import sys, os, time, json
import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from scipy import sparse
import shap

PROJECT_DIR = "/Users/namangarg/Desktop/KG-Trace/kg-trace"
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
CKPT = "/Users/namangarg/Desktop/KG-Trace/kg-trace/model/checkpoints/ablation_genomic_only/best.ckpt"

# load test ids
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
labels = test_data["labels"]

with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)

KMER_DIM = len(all_features)
NUM_GENES = 26

class GenomicOnly(nn.Module):
    def __init__(self, kmer_dim, num_genes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(kmer_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)

device = torch.device('cpu')
model = GenomicOnly(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_sample_indices = [sample_to_idx[s] for s in test_ids]
X_test = X_sparse[test_sample_indices].toarray().astype(np.float32)

res_idx = np.where(labels == 1)[0]
bg_x = torch.tensor(X_test[np.random.choice(len(X_test), 100, replace=False)])
explainer = shap.GradientExplainer(model, bg_x)

X_test_res = torch.tensor(X_test[res_idx])
shap_values = explainer.shap_values(X_test_res)

shap_vals = shap_values[1] # resistant class
mean_abs_shap = np.abs(shap_vals).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]

G = nx.read_graphml(os.path.join(KG_DIR, "amr_graph.graphml"))
DRUG = "INH"
cache = {}
def has_kg_path(feat_name):
    if feat_name in cache:
        return cache[feat_name]
    result = 0
    if G.has_node(feat_name) and G.has_node(DRUG) and nx.has_path(G, feat_name, DRUG):
        result = 1
    else:
        gene = feat_name.split(":")[0]
        if G.has_node(gene) and G.has_node(DRUG) and nx.has_path(G, gene, DRUG):
            result = 1
    cache[feat_name] = result
    return result

bgr_results = {}
for k in [10, 20, 50, 100]:
    top_k = [all_features[i] for i in sorted_indices[:k]]
    grounded = sum(has_kg_path(f) for f in top_k)
    bgr = grounded / k
    print(f"genomic_only BGR@{k}: {grounded}/{k} = {bgr:.2f}")

