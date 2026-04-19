#!/usr/bin/env python3
"""
Export the three ingredients needed for a SHAP beeswarm plot:
  1. SHAP Values Matrix  (samples × features)  → shap_values_top50.csv
  2. Original X_test     (samples × features)  → X_test_raw_top50.csv
  3. Feature Names       (list of strings)      → feature_names_top50.txt

Uses the EXACT same data sources as explain/shap_analysis.py:
  - test_ids from model/test_outputs.npz
  - sample list from features/mutation_samples.json
  - sparse feature matrix from features/mutation_matrix.npz
  - SHAP values from explain/shap_raw_values.npz
"""
import os, json
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(PROJECT_DIR, "explain", "final_publication_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load SHAP values and feature names ──
print("Loading SHAP values from explain/shap_raw_values.npz ...")
raw = np.load(os.path.join(PROJECT_DIR, "explain", "shap_raw_values.npz"), allow_pickle=True)
shap_vals = raw["shap_values"]        # shape: (n_test, n_features)
feat_names = raw["feature_names"]     # shape: (n_features,)
print(f"  SHAP matrix shape: {shap_vals.shape}")
print(f"  Total features:    {len(feat_names)}")

# ── 2. Identify Top 50 features by mean |SHAP| ──
print("\nRanking features by Mean |SHAP| ...")
mean_abs = np.mean(np.abs(shap_vals), axis=0)
top50_idx = np.argsort(mean_abs)[-50:][::-1]

shap_top50 = shap_vals[:, top50_idx]
names_top50 = feat_names[top50_idx]

print(f"  Top 5 features:")
for i in range(5):
    print(f"    {i+1}. {names_top50[i]}  (mean |SHAP| = {mean_abs[top50_idx[i]]:.6f})")

# ── 3. Load X_test (original feature values) ──
print("\nLoading X_test from features/mutation_matrix.npz ...")

# Get test sample IDs (same source as shap_analysis.py line 29)
test_data = np.load(os.path.join(PROJECT_DIR, "model", "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]

# Get full sample ordering (same source as shap_analysis.py line 93-94)
with open(os.path.join(PROJECT_DIR, "features", "mutation_samples.json")) as f:
    all_samples = json.load(f)

# Map test IDs to matrix row indices (same logic as shap_analysis.py line 106-108)
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_row_indices = [sample_to_idx[s] for s in test_ids]

# Load sparse matrix and extract test rows + top50 columns
X_sparse = sparse.load_npz(os.path.join(PROJECT_DIR, "features", "mutation_matrix.npz"))
X_test_top50 = X_sparse[test_row_indices][:, top50_idx].toarray()

print(f"  X_test shape (top 50): {X_test_top50.shape}")

# ── 4. Export everything to CSV ──
print("\n" + "=" * 60)

# 4a. SHAP Values Matrix
out_shap = os.path.join(OUT_DIR, "shap_values_top50.csv")
pd.DataFrame(shap_top50, columns=names_top50).to_csv(out_shap, index=False)
print(f"✓ SHAP Values Matrix → {out_shap}")
print(f"  ({shap_top50.shape[0]} samples × {shap_top50.shape[1]} features)")

# 4b. Original Feature Values (X_test)
out_xtest = os.path.join(OUT_DIR, "X_test_raw_top50.csv")
pd.DataFrame(X_test_top50, columns=names_top50).to_csv(out_xtest, index=False)
print(f"✓ Original X_test    → {out_xtest}")
print(f"  ({X_test_top50.shape[0]} samples × {X_test_top50.shape[1]} features)")

# 4c. Feature Names
out_names = os.path.join(OUT_DIR, "feature_names_top50.txt")
with open(out_names, "w") as f:
    for name in names_top50:
        f.write(f"{name}\n")
print(f"✓ Feature Names      → {out_names}")
print(f"  ({len(names_top50)} names)")

print("=" * 60)
print("\nDone! All three files are ready for beeswarm recreation.")
