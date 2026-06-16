"""
paths.py — Single source of truth for all paths in the KG-Trace pipeline.
Every script imports from here. No path is hardcoded anywhere else.

Layout assumed (configurable via environment variable KG_TRACE_BASE):
  <base>/
    KG-Trace/        ← this repository (kg-trace/ lives here)
    Mendeley Data/   ← downloaded Mendeley dataset
    Zenodo/          ← Zenodo data files

Set KG_TRACE_BASE to override the default (parent of this file's grandparent).
"""
import os

# Resolve base from repo location or environment override
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))   # kg-trace/
_REPO_DIR  = os.path.dirname(_THIS_DIR)                    # KG-Trace/
BASE_DIR   = os.environ.get("KG_TRACE_BASE", os.path.dirname(_REPO_DIR))

MENDELEY_DIR = os.path.join(BASE_DIR, "Mendeley Data")
ZENODO_DIR   = os.path.join(BASE_DIR, "Zenodo")
PROJECT_DIR  = _THIS_DIR  # kg-trace/

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
