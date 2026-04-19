#!/usr/bin/env python3
"""
Export the Attention Heatmap data:
  1. Attention Matrix  (samples × 26 genes)  → attention_matrix.csv
  2. Axis Labels       (26 gene names)        → attention_gene_labels.txt

Source: model/test_outputs.npz  (attn_weights, gene_names, labels)
"""
import os
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "explain", "final_publication_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load attention data ──
print("Loading attention weights from model/test_outputs.npz ...")
to = np.load(os.path.join(PROJECT_DIR, "model", "test_outputs.npz"), allow_pickle=True)

attn_weights = to["attn_weights"]   # shape: (n_test, 26)
gene_names = list(to["gene_names"]) # 26 gene names
labels = to["labels"]               # 0=Susceptible, 1=Resistant

print(f"  Attention matrix shape: {attn_weights.shape}")
print(f"  Genes (columns):        {len(gene_names)}")
print(f"  Gene names:             {gene_names}")
print(f"  Total samples:          {len(labels)}")
print(f"  Susceptible (0):        {int((labels == 0).sum())}")
print(f"  Resistant (1):          {int((labels == 1).sum())}")

# ── Sort samples by phenotype (Susceptible first, then Resistant) ──
order = np.argsort(labels)  # 0s first, then 1s
attn_sorted = attn_weights[order]
labels_sorted = labels[order]

# ── 1. Export Full Attention Matrix ──
out_full = os.path.join(OUT_DIR, "attention_matrix_full.csv")
df_full = pd.DataFrame(attn_sorted, columns=gene_names)
df_full.insert(0, "phenotype", ["Susceptible" if l == 0 else "Resistant" for l in labels_sorted])
df_full.to_csv(out_full, index=False)
print(f"\n✓ Full Attention Matrix → {out_full}")
print(f"  ({attn_sorted.shape[0]} samples × {attn_sorted.shape[1]} genes, sorted by phenotype)")

# ── 2. Export Subsampled Matrix (200 rows, as used in Figure 5) ──
n_show = 200
step = max(1, attn_sorted.shape[0] // n_show)
attn_sub = attn_sorted[::step][:n_show]
labels_sub = labels_sorted[::step][:n_show]

out_sub = os.path.join(OUT_DIR, "attention_matrix_200samples.csv")
df_sub = pd.DataFrame(attn_sub, columns=gene_names)
df_sub.insert(0, "phenotype", ["Susceptible" if l == 0 else "Resistant" for l in labels_sub])
df_sub.to_csv(out_sub, index=False)
print(f"✓ Subsampled Matrix    → {out_sub}")
print(f"  ({attn_sub.shape[0]} samples × {attn_sub.shape[1]} genes)")

# ── 3. Export Gene Labels ──
out_labels = os.path.join(OUT_DIR, "attention_gene_labels.txt")
with open(out_labels, "w") as f:
    for name in gene_names:
        f.write(f"{name}\n")
print(f"✓ Gene Labels          → {out_labels}")
print(f"  ({len(gene_names)} genes)")

# ── 4. Summary Statistics per Gene ──
out_stats = os.path.join(OUT_DIR, "attention_gene_stats.csv")
stats = []
for j, gene in enumerate(gene_names):
    s_vals = attn_weights[labels == 0, j]
    r_vals = attn_weights[labels == 1, j]
    stats.append({
        "gene": gene,
        "mean_all": attn_weights[:, j].mean(),
        "mean_susceptible": s_vals.mean(),
        "mean_resistant": r_vals.mean(),
        "diff_R_minus_S": r_vals.mean() - s_vals.mean(),
    })
df_stats = pd.DataFrame(stats).sort_values("mean_all", ascending=False)
df_stats.to_csv(out_stats, index=False, float_format="%.6f")
print(f"✓ Gene Stats           → {out_stats}")

print("\n" + "=" * 60)
print("Done! All attention heatmap data exported.")
