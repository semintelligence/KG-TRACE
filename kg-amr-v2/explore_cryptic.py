"""
Step 1a: Explore CRyPTIC data schemas to understand exact column names/types.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from paths import *

import pandas as pd
import pickle, gzip

print("="*60)
print("1. MUTATIONS.parquet")
print("="*60)
mut = pd.read_parquet(os.path.join(ZENODO_DIR, "MUTATIONS.parquet"))
print(f"Shape: {mut.shape}")
print(f"Columns: {list(mut.columns)}")
print(f"Index names: {mut.index.names}")
if isinstance(mut.index, pd.MultiIndex):
    print(f"Index levels: {mut.index.nlevels}")
    for i, name in enumerate(mut.index.names):
        print(f"  Level {i} ({name}): {mut.index.get_level_values(i).nunique()} unique")
print(f"\nHead:\n{mut.head(10)}")
print(f"\nDtypes:\n{mut.dtypes}")

print("\n" + "="*60)
print("2. EFFECTS.parquet")
print("="*60)
eff = pd.read_parquet(os.path.join(ZENODO_DIR, "EFFECTS.parquet"))
print(f"Shape: {eff.shape}")
print(f"Columns: {list(eff.columns)}")
print(f"Index names: {eff.index.names}")
if isinstance(eff.index, pd.MultiIndex):
    for i, name in enumerate(eff.index.names):
        print(f"  Level {i} ({name}): {eff.index.get_level_values(i).nunique()} unique")
print(f"\nHead:\n{eff.head(10)}")
print(f"\nDtypes:\n{eff.dtypes}")

print("\n" + "="*60)
print("3. DST_MEASUREMENTS.pkl.gz")
print("="*60)
with gzip.open(os.path.join(ZENODO_DIR, "DST_MEASUREMENTS.pkl.gz"), "rb") as f:
    dst = pickle.load(f)
print(f"Type: {type(dst)}")
if isinstance(dst, pd.DataFrame):
    print(f"Shape: {dst.shape}")
    print(f"Columns: {list(dst.columns)}")
    print(f"Index names: {dst.index.names}")
    if isinstance(dst.index, pd.MultiIndex):
        for i, name in enumerate(dst.index.names):
            print(f"  Level {i} ({name}): {dst.index.get_level_values(i).nunique()} unique")
    print(f"\nHead:\n{dst.head(10)}")
    print(f"\nDtypes:\n{dst.dtypes}")
    # Check PHENOTYPE distribution
    if 'PHENOTYPE' in dst.columns:
        print(f"\nPHENOTYPE distribution:\n{dst['PHENOTYPE'].value_counts()}")
    # Check drugs
    if 'DRUG' in dst.index.names:
        drug_idx = dst.index.get_level_values('DRUG')
        print(f"\nDrug counts:\n{drug_idx.value_counts().head(20)}")
elif isinstance(dst, dict):
    print(f"Keys: {list(dst.keys())[:20]}")
    first_key = list(dst.keys())[0]
    print(f"First value type: {type(dst[first_key])}")
    print(f"First value: {dst[first_key]}")

print("\n" + "="*60)
print("4. GENOMES.parquet")
print("="*60)
gen = pd.read_parquet(os.path.join(ZENODO_DIR, "GENOMES.parquet"))
print(f"Shape: {gen.shape}")
print(f"Columns: {list(gen.columns)}")
print(f"Index names: {gen.index.names}")
print(f"\nHead:\n{gen.head(5)}")
print(f"\nDtypes:\n{gen.dtypes}")

print("\n" + "="*60)
print("5. WGS_SAMPLES.parquet")
print("="*60)
wgs = pd.read_parquet(os.path.join(ZENODO_DIR, "WGS_SAMPLES.parquet"))
print(f"Shape: {wgs.shape}")
print(f"Columns: {list(wgs.columns)}")
print(f"\nHead:\n{wgs.head(5)}")

print("\n" + "="*60)
print("6. PREDICTIONS.parquet")
print("="*60)
pred = pd.read_parquet(os.path.join(ZENODO_DIR, "PREDICTIONS.parquet"))
print(f"Shape: {pred.shape}")
print(f"Columns: {list(pred.columns)}")
print(f"Index names: {pred.index.names}")
print(f"\nHead:\n{pred.head(5)}")
print(f"\nDtypes:\n{pred.dtypes}")

print("\nDONE — schema exploration complete")
