#!/usr/bin/env python3
"""
run_mendeley_pipeline.py — Master runner for Mendeley multi-species extension.
Executes stages M1-M8 in sequence, skipping already-completed stages.

Usage:
  python3 run_mendeley_pipeline.py              # run all stages
  python3 run_mendeley_pipeline.py --from m3    # resume from stage M3
  python3 run_mendeley_pipeline.py --only m3    # run only stage M3
"""
import os, sys, argparse, subprocess, time
from pathlib import Path

# cd into project root
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

STAGES = [
    ("m1", "stage_m1_fasta_qc.py",        "FASTA Quality Control"),
    ("m2", "stage_m2_kmer_features.py",    "K-mer TF-IDF Extraction"),
    ("m3", "stage_m3_bvbrc_annotations.py","BV-BRC AMR Annotations"),
    ("m4", "stage_m4_labels.py",           "Label Alignment"),
    ("m5", "stage_m5_species_kg.py",       "Species KG + RotatE"),
    ("m6", "stage_m6_train_species.py",    "Train Per-Species Models"),
    ("m7", "stage_m7_explain_species.py",  "Per-Species Explainability"),
    ("m8", "stage_m8_results.py",          "Results + Dashboard Update"),
]

# Completion sentinel files (stage considered done if these exist)
COMPLETION_CHECKS = {
    "m1": "data/download_logs/Ecoli_ampicillin_valid_fastas.txt",
    "m2": "features/matrices/Ecoli_ampicillin_kmer.npz",
    "m3": "features/annotations/Ecoli_ampicillin_amr_raw.json",
    "m4": "features/labels/Ecoli_ampicillin_labels.csv",
    "m5": "kg/species/Ecoli_ampicillin/entity_embeddings.npy",
    "m6": "model/species/Ecoli_ampicillin/test_results.json",
    "m7": "explain/species/Ecoli_ampicillin/shap_values.csv",
    "m8": "evaluate/final_results.csv",
}


def is_done(stage_key):
    check = COMPLETION_CHECKS.get(stage_key)
    return check and os.path.exists(os.path.join(PROJECT_DIR, check))


def run_stage(script_path, stage_name):
    print(f"\n{'#'*70}")
    print(f"# STAGE: {stage_name}")
    print(f"# Script: {script_path}")
    print(f"{'#'*70}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=PROJECT_DIR,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAIL] Stage exited with code {result.returncode} "
              f"after {elapsed:.0f}s")
        sys.exit(result.returncode)
    print(f"\n[DONE] {stage_name} completed in {elapsed:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_stage", default=None,
                        help="Start from this stage (e.g. m3)")
    parser.add_argument("--only", dest="only_stage", default=None,
                        help="Run only this stage (e.g. m3)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run even if stage is marked complete")
    args = parser.parse_args()

    start_idx = 0
    if args.from_stage:
        keys = [s[0] for s in STAGES]
        if args.from_stage not in keys:
            print(f"Unknown stage: {args.from_stage}. Valid: {keys}")
            sys.exit(1)
        start_idx = keys.index(args.from_stage)

    print("=" * 70)
    print("  KG-AMR — Mendeley Multi-Species Pipeline")
    print("=" * 70)
    print(f"  Working dir: {PROJECT_DIR}")
    print(f"  Python:      {sys.executable}")
    print(f"  Stages:      {len(STAGES)}")
    print()

    total_t0 = time.time()
    completed = []

    for key, script, name in STAGES[start_idx:]:
        if args.only_stage and key != args.only_stage:
            continue
        if not args.force and is_done(key):
            print(f"  [SKIP] {key}: {name} — already complete")
            completed.append((key, name, "skipped"))
            continue
        run_stage(script, name)
        completed.append((key, name, "ran"))

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  Pipeline complete in {total_elapsed/60:.1f} min")
    for key, name, status in completed:
        mark = "✓" if status == "ran" else "↩"
        print(f"    {mark}  {key}  {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
