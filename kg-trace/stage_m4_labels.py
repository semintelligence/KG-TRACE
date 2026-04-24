"""
Stage M4: Label Alignment
Loads Mendeley phenotype labels and aligns them to valid FASTAs.

Mendeley TXT format (tab-separated, with leading index column):
    <index>  genome_id  resistant_phenotype

For carbapenem species, merges two drugs (meropenem + imipenem):
  - 1 = resistant (union: resistant to either drug)
  - 0 = susceptible (susceptible to both)
  - Ties broken by majority; discordant pairs assigned the max label (conservative).

Outputs per species:
  features/labels/{species}_labels.csv  (genome_id, label columns)
"""
import os, sys, json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR, MENDELEY_DIR as _MENDELEY_BASE

MENDELEY_DIR = os.path.join(_MENDELEY_BASE, "genome_id")
LOG_DIR      = os.path.join(PROJECT_DIR, "data/download_logs")
LABELS_DIR   = os.path.join(PROJECT_DIR, "features/labels")
os.makedirs(LABELS_DIR, exist_ok=True)

# Species → Mendeley TXT filename(s)
SPECIES_CONFIGS = {
    "Ecoli_ampicillin": [
        "Data_Escherichia_coli_ampicillin.txt",
    ],
    "Kpneumoniae_cipro": [
        "Data_Klebsiella_pneumoniae_ciprofloxacin.txt",
    ],
    "Kpneumoniae_carbapenem": [
        "Data_Klebsiella_pneumoniae_meropenem.txt",
        "Data_Klebsiella_pneumoniae_imipenem.txt",
    ],
    "Abaumannii_carbapenem": [
        "Data_Acinetobacter_baumannii_meropenem.txt",
        "Data_Acinetobacter_baumannii_imipenem.txt",
    ],
}

def load_mendeley_txt(path):
    """Load a Mendeley genome_id phenotype file (tab-sep with leading index column)."""
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # Columns: first might be row-index, then genome_id, then resistant_phenotype
    df.columns = df.columns.str.strip()
    if "genome_id" not in df.columns or "resistant_phenotype" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    df["genome_id"] = df["genome_id"].astype(str).str.strip()
    df["resistant_phenotype"] = df["resistant_phenotype"].astype(int)
    return df[["genome_id", "resistant_phenotype"]].drop_duplicates("genome_id")


summary = {}

for species, txt_files in SPECIES_CONFIGS.items():
    print(f"\n{species}")

    # Load valid genome IDs from Stage M1
    valid_file = os.path.join(LOG_DIR, f"{species}_valid_fastas.txt")
    if not os.path.exists(valid_file):
        print(f"  [SKIP] No valid FASTA list")
        continue
    with open(valid_file) as f:
        valid_ids = set(l.strip() for l in f if l.strip())
    print(f"  Valid FASTAs: {len(valid_ids)}")

    # Load and merge label files
    dfs = []
    for txt_name in txt_files:
        txt_path = os.path.join(MENDELEY_DIR, txt_name)
        if not os.path.exists(txt_path):
            print(f"  [WARN] Not found: {txt_name}")
            continue
        df_part = load_mendeley_txt(txt_path)
        print(f"  Loaded {len(df_part):4d} labels from {txt_name}")
        dfs.append(df_part)

    if not dfs:
        print(f"  [SKIP] No label files found")
        continue

    if len(dfs) == 1:
        labels_df = dfs[0]
    else:
        # Merge multiple drugs: conservative union
        # Genome IDs shared across files: keep max label (resistant wins)
        merged = dfs[0].copy()
        for extra in dfs[1:]:
            merged = merged.merge(extra, on="genome_id",
                                  suffixes=("_a", "_b"), how="outer")
            cols = [c for c in merged.columns if c != "genome_id"]
            merged["resistant_phenotype"] = merged[cols].max(axis=1).fillna(0).astype(int)
            merged = merged[["genome_id", "resistant_phenotype"]]
        labels_df = merged.drop_duplicates("genome_id")

    labels_df = labels_df[labels_df["genome_id"].isin(valid_ids)]
    print(f"  After filtering to valid FASTAs: {len(labels_df)}")
    print(f"  Label distribution: {(labels_df['resistant_phenotype']==1).sum()} R "
          f"/ {(labels_df['resistant_phenotype']==0).sum()} S "
          f"({100*(labels_df['resistant_phenotype']==1).mean():.1f}% R)")

    if len(labels_df) < 50:
        print(f"  [WARN] Very few labelled genomes — check label file paths")

    out_path = os.path.join(LABELS_DIR, f"{species}_labels.csv")
    labels_df.to_csv(out_path, index=False)
    print(f"  → {out_path}")

    summary[species] = {
        "n_labeled": len(labels_df),
        "n_R": int((labels_df["resistant_phenotype"] == 1).sum()),
        "n_S": int((labels_df["resistant_phenotype"] == 0).sum()),
    }

print("\n=== Label Summary ===")
for sp, s in summary.items():
    pct = 100 * s["n_R"] / s["n_labeled"] if s["n_labeled"] else 0
    print(f"  {sp:<28}  {s['n_labeled']:4d} total  "
          f"{s['n_R']:4d} R  {s['n_S']:4d} S  ({pct:.1f}% R)")
print("\nStage M4 complete.")
