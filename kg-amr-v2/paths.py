"""
paths.py — Single source of truth for all paths in the KG-AMR pipeline.
Every script imports from here. No path is hardcoded anywhere else.
"""
import os

BASE_DIR     = os.path.expanduser("~/Desktop/AMR NamanXSarika")
MENDELEY_DIR = os.path.join(BASE_DIR, "Mendeley Data")
ZENODO_DIR   = os.path.join(BASE_DIR, "Zenodo")
PROJECT_DIR  = os.path.join(BASE_DIR, "KG-AMR")

# ── Model hyper-parameters (single source of truth) ────────────────────────
KG_EMBED_DIM   = 64   # RotatE entity/relation embedding dimension
GENOMIC_HIDDEN = 128   # Genomic encoder hidden dimension
FUSED_DIM      = 128   # Dimension after cross-attention fusion

# Verify data dirs exist (project dir is created by us)
for path in [MENDELEY_DIR, ZENODO_DIR]:
    assert os.path.exists(path), f"ABORT: Path not found: {path}"
os.makedirs(PROJECT_DIR, exist_ok=True)


def guard():
    """Call at top of every script to verify we're in the right place."""
    cwd = os.getcwd()
    assert BASE_DIR in cwd or PROJECT_DIR in cwd, (
        f"ABORT: Must run from inside {BASE_DIR}, got {cwd}"
    )
    print(f"[paths] Mendeley: {MENDELEY_DIR}")
    print(f"[paths] Zenodo:   {ZENODO_DIR}")
    print(f"[paths] Project:  {PROJECT_DIR}")
