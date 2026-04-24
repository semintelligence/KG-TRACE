"""
Step 10: Generate final results table — evaluate/final_results.csv
All values from actual computed outputs, no hardcoding.
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np

EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
os.makedirs(EVALUATE_DIR, exist_ok=True)
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
BASELINES_DIR = os.path.join(PROJECT_DIR, "baselines")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")

# ── Load all results ────────────────────────────────────────────────────────
print("[1/2] Loading results...")

# KG-Trace
with open(os.path.join(MODEL_DIR, "test_results.json")) as f:
    kg = json.load(f)

# Baselines
with open(os.path.join(BASELINES_DIR, "baseline_results.json")) as f:
    baselines = json.load(f)

# Alignment metrics (KG-Trace only)
with open(os.path.join(EXPLAIN_DIR, "alignment_metrics.json")) as f:
    alignment = json.load(f)

# Test outputs (for gate)
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
gate_values = test_data["gate_values"]
gate_mean = float(gate_values.mean(axis=1).mean())

# Pathway coverage
pw_path = os.path.join(EXPLAIN_DIR, "pathway_explanations.json")
pathway_coverage = "n/a"
if os.path.exists(pw_path):
    try:
        with open(pw_path) as f:
            pw = json.load(f)
        # Count resistant test genomes with at least 1 valid path
        labels = test_data["labels"]
        preds = test_data["preds"]
        test_ids = list(test_data["test_ids"])
        n_resistant = int((labels == 1).sum())
        n_with_path = 0
        for i, tid in enumerate(test_ids):
            if labels[i] == 1 and str(tid) in pw:
                paths = pw[str(tid)]
                if isinstance(paths, list) and len(paths) > 0:
                    n_with_path += 1
                elif isinstance(paths, dict) and paths:
                    n_with_path += 1
        pathway_coverage = f"{n_with_path}/{n_resistant} ({100*n_with_path/n_resistant:.1f}%)" if n_resistant > 0 else "0"
    except Exception as e:
        pathway_coverage = f"Error: {e}"

spearman_attn_shap = alignment.get("spearman_attn_vs_shap", {})
spearman_shap_card = alignment.get("spearman_shap_vs_card", {})

# ── Build table ─────────────────────────────────────────────────────────────
print("[2/2] Building final results table...")

rows = []

# KG-Trace
rows.append({
    "Model": "KG-Trace",
    "AUROC": f"{kg['auroc']:.4f}",
    "F1-macro": f"{kg['f1_macro']:.4f}",
    "Precision": f"{kg['precision_macro']:.4f}",
    "Recall": f"{kg['recall_macro']:.4f}",
    "BCS@10": f"{alignment.get('bcs_global_10', 'n/a')}",
    "Spearman_rho": f"{spearman_attn_shap.get('rho', 'n/a'):.4f}",
    "Spearman_pvalue": f"{spearman_attn_shap.get('pvalue', 'n/a'):.6f}",
    "Gate_mean": f"{gate_mean:.4f}",
    "Pathway_coverage": pathway_coverage,
})

# Baselines
for b in baselines:
    bcs = b.get("bcs_10")
    rows.append({
        "Model": b["model"],
        "AUROC": f"{b['auroc']:.4f}",
        "F1-macro": f"{b['f1_macro']:.4f}",
        "Precision": f"{b['precision_macro']:.4f}",
        "Recall": f"{b['recall_macro']:.4f}",
        "BCS@10": f"{bcs:.2f}" if bcs is not None else "n/a",
        "Spearman_rho": "n/a",
        "Spearman_pvalue": "n/a",
        "Gate_mean": "n/a",
        "Pathway_coverage": "n/a",
    })

# Write CSV
csv_path = os.path.join(EVALUATE_DIR, "final_results.csv")
fieldnames = ["Model", "AUROC", "F1-macro", "Precision", "Recall", "BCS@10",
              "Spearman_rho", "Spearman_pvalue", "Gate_mean", "Pathway_coverage"]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

# Also write JSON
json_path = os.path.join(EVALUATE_DIR, "final_results.json")
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

# Print table
print(f"\n{'Model':<15s} {'AUROC':>8s} {'F1-macro':>10s} {'BCS@10':>8s} {'Spearman_rho':>14s} {'Gate':>6s}")
print("-" * 65)
for r in rows:
    print(f"{r['Model']:<15s} {r['AUROC']:>8s} {r['F1-macro']:>10s} {r['BCS@10']:>8s} {r['Spearman_rho']:>14s} {r['Gate_mean']:>6s}")

print(f"\n  HONEST COMPARISON:")
print(f"  SVM AUROC ({baselines[0]['auroc']:.4f}) > KG-Trace AUROC ({kg['auroc']:.4f})")
print(f"  KG-Trace provides explainability via attention + KG pathways that baselines cannot.")
print(f"\n  Saved to: {csv_path}")
print(f"  Saved to: {json_path}")
print("DONE — generate_final_results.py")
