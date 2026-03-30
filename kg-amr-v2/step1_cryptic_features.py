"""
Step 1: CRyPTIC Feature Extraction
- Build binary mutation matrix [sample × gene_mutation] from MUTATIONS.parquet
- Extract R/S labels from DST_MEASUREMENTS.pkl.gz
- Focus on catalogue genes (26 genes from EFFECTS) for tractable feature space
- Save sparse matrices + label DataFrames per drug
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from paths import *

import pandas as pd
import numpy as np
from scipy import sparse
import pickle, gzip, json

OUT_DIR = os.path.join(PROJECT_DIR, "features")
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

# ── 1. Load EFFECTS to get catalogue genes ──────────────────────────────────
print("[1/5] Loading EFFECTS.parquet to get catalogue genes...")
effects = pd.read_parquet(os.path.join(ZENODO_DIR, "EFFECTS.parquet"))
catalogue_genes = sorted([g for g in effects.index.get_level_values("GENE").unique() if isinstance(g, str)])
print(f"  Catalogue genes ({len(catalogue_genes)}): {catalogue_genes}")

# Also extract the drug list from EFFECTS
catalogue_drugs = sorted([d for d in effects.index.get_level_values("DRUG").unique() if isinstance(d, str)])
print(f"  Catalogue drugs ({len(catalogue_drugs)}): {catalogue_drugs}")

# ── 2. Load MUTATIONS, filter to catalogue genes ────────────────────────────
print("\n[2/5] Loading MUTATIONS.parquet (filtering to catalogue genes)...")
# Read only the index — no columns needed, we just need (UNIQUEID, GENE, MUTATION) presence
mut = pd.read_parquet(
    os.path.join(ZENODO_DIR, "MUTATIONS.parquet"),
    columns=["IS_NULL"]  # smallest column to load
)
print(f"  Full mutations: {len(mut):,} rows, {mut.index.get_level_values('UNIQUEID').nunique():,} samples")

# Filter to catalogue genes
gene_level = mut.index.get_level_values("GENE")
mask = gene_level.isin(catalogue_genes)
mut_filtered = mut.loc[mask]
del mut  # free memory

# Remove null calls (no mutation actually present)
if "IS_NULL" in mut_filtered.columns:
    mut_filtered = mut_filtered[~mut_filtered["IS_NULL"]]

print(f"  After catalogue gene filter + non-null: {len(mut_filtered):,} rows")

# Create gene_mutation feature name
sample_ids = mut_filtered.index.get_level_values("UNIQUEID")
gene_names = mut_filtered.index.get_level_values("GENE")
mutation_names = mut_filtered.index.get_level_values("MUTATION")
features = gene_names.astype(str) + ":" + mutation_names.astype(str)

# Get unique feature names and sample IDs
unique_samples = sorted(sample_ids.unique().tolist())
unique_features = sorted(features.unique().tolist())
print(f"  Unique samples in mutations: {len(unique_samples):,}")
print(f"  Unique gene:mutation features: {len(unique_features):,}")

# ── 3. Build sparse binary matrix ───────────────────────────────────────────
print("\n[3/5] Building sparse binary mutation matrix...")
sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
feature_to_idx = {f: i for i, f in enumerate(unique_features)}

row_indices = np.array([sample_to_idx[s] for s in sample_ids], dtype=np.int32)
col_indices = np.array([feature_to_idx[f] for f in features], dtype=np.int32)
data = np.ones(len(row_indices), dtype=np.int8)

X = sparse.csr_matrix(
    (data, (row_indices, col_indices)),
    shape=(len(unique_samples), len(unique_features)),
    dtype=np.int8
)
# Remove duplicates (some samples may have same gene:mutation listed more than once)
X.data[:] = 1

del row_indices, col_indices, data, sample_ids, gene_names, mutation_names, features, mut_filtered

print(f"  Matrix shape: {X.shape}")
print(f"  Non-zeros: {X.nnz:,} ({100*X.nnz/(X.shape[0]*X.shape[1]):.4f}% dense)")
print(f"  Memory: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)/1e6:.1f} MB")

# Save matrix and metadata
sparse.save_npz(os.path.join(OUT_DIR, "mutation_matrix.npz"), X)
with open(os.path.join(OUT_DIR, "mutation_samples.json"), "w") as f:
    json.dump(unique_samples, f)
with open(os.path.join(OUT_DIR, "mutation_features.json"), "w") as f:
    json.dump(unique_features, f)
print(f"  Saved mutation_matrix.npz, mutation_samples.json, mutation_features.json")

# ── 4. Extract R/S labels per drug ──────────────────────────────────────────
print("\n[4/5] Loading DST_MEASUREMENTS and extracting labels...")
with gzip.open(os.path.join(ZENODO_DIR, "DST_MEASUREMENTS.pkl.gz"), "rb") as f:
    dst = pickle.load(f)

# Filter to R/S only (drop U=unknown, I=intermediate)
dst_rs = dst[dst["PHENOTYPE"].isin(["R", "S"])].copy()
print(f"  Total R/S measurements: {len(dst_rs):,}")

# For each drug, create label series aligned with mutation matrix samples
labels_dir = os.path.join(OUT_DIR, "labels")
os.makedirs(labels_dir, exist_ok=True)

sample_set = set(unique_samples)
drug_stats = {}

for drug in catalogue_drugs:
    drug_data = dst_rs.xs(drug, level="DRUG") if drug in dst_rs.index.get_level_values("DRUG").unique() else pd.DataFrame()
    if len(drug_data) == 0:
        print(f"  {drug}: no R/S data, skipping")
        continue
    
    # Intersect with mutation matrix samples
    common = drug_data.index.intersection(sample_set)
    if len(common) == 0:
        print(f"  {drug}: no overlap with mutation samples, skipping")
        continue
    
    labels = drug_data.loc[common, "PHENOTYPE"].map({"R": 1, "S": 0})
    # Drop any NaN (shouldn't happen but safety)
    labels = labels.dropna().astype(int)
    
    n_r = (labels == 1).sum()
    n_s = (labels == 0).sum()
    drug_stats[drug] = {"total": len(labels), "R": int(n_r), "S": int(n_s), 
                         "R_pct": round(100*n_r/len(labels), 1)}
    
    labels.to_frame("label").to_parquet(os.path.join(labels_dir, f"labels_{drug}.parquet"))
    print(f"  {drug}: {len(labels):,} samples ({n_r:,} R / {n_s:,} S = {100*n_r/len(labels):.1f}% R)")

# ── 5. Summary ──────────────────────────────────────────────────────────────
print("\n[5/5] Summary")
elapsed = time.time() - t0
summary = {
    "matrix_shape": list(X.shape),
    "nnz": int(X.nnz),
    "density_pct": round(100*X.nnz/(X.shape[0]*X.shape[1]), 4),
    "n_catalogue_genes": len(catalogue_genes),
    "catalogue_genes": catalogue_genes,
    "drug_stats": drug_stats,
    "elapsed_seconds": round(elapsed, 1)
}
with open(os.path.join(OUT_DIR, "feature_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Matrix: {X.shape[0]} samples × {X.shape[1]} features")
print(f"  Drugs with labels: {len(drug_stats)}")
print(f"  Elapsed: {elapsed:.1f}s")
print(f"  All outputs in: {OUT_DIR}")
print("\nDONE")
