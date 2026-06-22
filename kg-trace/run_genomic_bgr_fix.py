import os
import sys
import torch
import torch.nn as nn
import json
import shap
import numpy as np

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
        self.gene_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_genes)
        )
    def forward(self, x):
        h = self.encoder(x)
        return self.classifier(h)

device = torch.device('cpu')
features_json = "/Users/namangarg/Desktop/KG-Trace/kg-trace/features/mutation_features.json"
with open(features_json, 'r') as f:
    feature_names = json.load(f)

KMER_DIM = len(feature_names)
NUM_GENES = 26

model = GenomicOnly(kmer_dim=KMER_DIM, num_genes=NUM_GENES)
ckpt = torch.load("/Users/namangarg/Desktop/KG-Trace/kg-trace/model/checkpoints/ablation_genomic_only/best.ckpt", map_location=device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

test_x = torch.load("/Users/namangarg/Desktop/KG-Trace/kg-trace/features/INH_test_x.pt")
test_y = torch.load("/Users/namangarg/Desktop/KG-Trace/kg-trace/features/INH_test_y.pt")
res_idx = (test_y == 1).nonzero(as_tuple=True)[0]
bg_x = test_x[np.random.choice(len(test_x), 100, replace=False)].float()
explainer = shap.GradientExplainer(model, bg_x)

test_x_res = test_x[res_idx].float()
shap_values = explainer.shap_values(test_x_res)

shap_vals = shap_values[1]
mean_shap = np.abs(shap_vals).mean(axis=0)
top_50_idx = np.argsort(mean_shap)[-50:][::-1]

kg_triples_file = "/Users/namangarg/Desktop/KG-Trace/kg-trace/kg_data/mtb_kg_triples.json"
import collections
with open(kg_triples_file, 'r') as f:
    triples = json.load(f)
mut_to_drug = collections.defaultdict(set)
for head, rel, tail in triples:
    if rel in ["confers_resistance_to", "confers_susceptibility_to", "has_uncertain_effect_on"]:
        mut_to_drug[head].add(tail)

hits = [0, 0, 0]
drug_node = "Isoniazid"
for rank, idx in enumerate(top_50_idx):
    mut_name = feature_names[idx]
    has_path = (drug_node in mut_to_drug.get(mut_name, set()))
    if has_path:
        if rank < 10: hits[0] += 1
        if rank < 20: hits[1] += 1
        if rank < 50: hits[2] += 1

print(f"genomic_only BGR@10: {hits[0]/10:.2f}")
print(f"genomic_only BGR@20: {hits[1]/20:.2f}")
print(f"genomic_only BGR@50: {hits[2]/50:.2f}")
