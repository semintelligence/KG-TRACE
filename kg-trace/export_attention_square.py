#!/usr/bin/env python3
"""
Export a 26×26 square attention matrix for heatmap visualization.

Since KG-TRACE uses cross-attention (producing a [n_test, 26] vector per sample,
NOT a 26×26 self-attention matrix), we construct the square matrix as follows:

  Method: Pearson correlation of attention patterns across all test samples.
  Entry (i, j) = correlation between gene_i and gene_j attention weights
                  across all 5,665 test isolates.

This captures which genes are consistently co-attended by the model —
a standard representation for cross-attention architectures.

Outputs:
  1. attention_26x26_matrix.csv   — the 26×26 square matrix
  2. attention_gene_labels.txt    — 26 gene name labels (rows & columns)
  3. attention_26x26_mean.csv     — mean attention per gene (for bar chart)
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
attn = to["attn_weights"]       # shape: (5665, 26)
gene_names = list(to["gene_names"])
labels = to["labels"]

print(f"  Raw attention shape: {attn.shape}  (samples × genes)")
print(f"  Gene names: {gene_names}")

# ── Build 26×26 co-attention correlation matrix ──
print("\nComputing 26×26 Pearson correlation of attention across all test samples...")
corr_matrix = np.corrcoef(attn.T)  # Transpose: (26, n_test) → corrcoef gives (26, 26)
print(f"  Correlation matrix shape: {corr_matrix.shape}")

# ── Also compute separate matrices for Susceptible vs Resistant ──
attn_S = attn[labels == 0]
attn_R = attn[labels == 1]
corr_S = np.corrcoef(attn_S.T)
corr_R = np.corrcoef(attn_R.T)

# ── Export 1: Full 26×26 matrix ──
out1 = os.path.join(OUT_DIR, "attention_26x26_matrix.csv")
df = pd.DataFrame(corr_matrix, index=gene_names, columns=gene_names)
df.to_csv(out1)
print(f"\n✓ 26×26 Attention Matrix (All)          → {out1}")

# ── Export 2: Susceptible-only 26×26 ──
out2 = os.path.join(OUT_DIR, "attention_26x26_susceptible.csv")
pd.DataFrame(corr_S, index=gene_names, columns=gene_names).to_csv(out2)
print(f"✓ 26×26 Attention Matrix (Susceptible)  → {out2}")

# ── Export 3: Resistant-only 26×26 ──
out3 = os.path.join(OUT_DIR, "attention_26x26_resistant.csv")
pd.DataFrame(corr_R, index=gene_names, columns=gene_names).to_csv(out3)
print(f"✓ 26×26 Attention Matrix (Resistant)    → {out3}")

# ── Export 4: Gene labels ──
out4 = os.path.join(OUT_DIR, "attention_gene_labels.txt")
with open(out4, "w") as f:
    for name in gene_names:
        f.write(f"{name}\n")
print(f"✓ Gene Labels                           → {out4}")

# ── Export 5: Mean attention per gene ──
out5 = os.path.join(OUT_DIR, "attention_26x26_mean.csv")
mean_all = attn.mean(axis=0)
mean_S = attn_S.mean(axis=0)
mean_R = attn_R.mean(axis=0)
df_mean = pd.DataFrame({
    "gene": gene_names,
    "mean_attn_all": mean_all,
    "mean_attn_susceptible": mean_S,
    "mean_attn_resistant": mean_R,
    "diff_R_minus_S": mean_R - mean_S,
})
df_mean = df_mean.sort_values("mean_attn_all", ascending=False)
df_mean.to_csv(out5, index=False, float_format="%.6f")
print(f"✓ Mean Attention per Gene               → {out5}")

# ── Print preview ──
print("\n" + "=" * 60)
print("Preview of 26×26 matrix (first 5×5 corner):")
print(df.iloc[:5, :5].to_string(float_format="%.4f"))
print("\nTop 5 genes by mean attention:")
for _, row in df_mean.head(5).iterrows():
    print(f"  {row['gene']:12s}  all={row['mean_attn_all']:.4f}  S={row['mean_attn_susceptible']:.4f}  R={row['mean_attn_resistant']:.4f}")
print("=" * 60)
print("\nDone! All 26×26 heatmap data exported.")
