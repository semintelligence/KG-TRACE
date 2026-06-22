#!/bin/bash
set -e

echo "Starting INH Retraining Pipeline..."

echo "[1/6] Training KG-Trace on INH..."
python model/train.py

echo "[2/6] Training Genomic-Only Baseline on INH..."
python evaluate/run_ablation.py

echo "[3/6] Computing SHAP for KG-Trace..."
python explain/shap_analysis.py

echo "[4/6] Computing SHAP for Genomic-Only Baseline..."
python explain/shap_genomic_only.py

echo "[5/6] Computing Alignment Metrics (BCS/BGR)..."
python explain/alignment_metrics.py

echo "[6/6] Generating Final Results..."
python evaluate/generate_final_results.py

echo "ALL DONE!"
