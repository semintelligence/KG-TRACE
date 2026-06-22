#!/bin/bash
set -e

echo "Starting RIF Retraining Pipeline..."

echo "[1/7] Training KG-Trace on RIF..."
python model/train.py

echo "[2/7] Training Genomic-Only Baseline on RIF..."
python evaluate/run_ablation.py

echo "[3/7] Computing SHAP for KG-Trace..."
python explain/shap_analysis.py

echo "[4/7] Computing SHAP for Genomic-Only Baseline..."
python explain/shap_genomic_only.py

echo "[5/7] Computing Alignment Metrics (BCS/BGR)..."
python explain/alignment_metrics.py

echo "[6/7] Computing Pathway Coverage..."
python explain/pathway_explain_fixed.py

echo "[7/7] Generating Final Results..."
python evaluate/generate_final_results.py

echo "ALL DONE!"
