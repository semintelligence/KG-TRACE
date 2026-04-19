"""
Compile multi-dataset results from all evaluations into a single table.
Combines: baseline INH (100-epoch), MTB extension drugs (10-epoch),
and ablation study results.
"""
import os, json, csv
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

# ── 1. Baseline INH results (100-epoch trained model) ───────────────────────
with open(os.path.join(MODEL_DIR, "test_results.json")) as f:
    inh_results = json.load(f)

# ── 2. MTB extension results ────────────────────────────────────────────────
with open(os.path.join(EVALUATE_DIR, "mtb_extension_results.json")) as f:
    mtb_results = json.load(f)

# ── 3. Ablation results ─────────────────────────────────────────────────────
with open(os.path.join(EVALUATE_DIR, "ablation_results.json")) as f:
    ablation_results = json.load(f)

# ── 4. Mendeley data report ─────────────────────────────────────────────────
with open(os.path.join(EVALUATE_DIR, "mendeley_data_report.json")) as f:
    mendeley_report = json.load(f)

# ── Build multi-dataset table ───────────────────────────────────────────────

rows = []

# Baseline INH
rows.append({
    "Dataset": "MTB_INH",
    "Species": "M. tuberculosis",
    "Drug": "INH (Isoniazid)",
    "Model": "KG-AMR (100 epochs)",
    "AUROC": inh_results["auroc"],
    "F1_macro": inh_results["f1_macro"],
    "N_samples": inh_results["n_test_R"] + inh_results["n_test_S"],
    "N_test": inh_results["n_test"],
    "Note": "Baseline model",
})

# MTB extensions
for r in mtb_results:
    rows.append({
        "Dataset": r["dataset"],
        "Species": r["species"],
        "Drug": r["drug"],
        "Model": "KG-AMR (10 epochs)",
        "AUROC": r["auroc"],
        "F1_macro": r["f1_macro"],
        "N_samples": r["n_samples"],
        "N_test": r["n_test"],
        "Note": "Extension",
    })

# Mendeley datasets (labels only — no model trained)
for p in mendeley_report["pairs"]:
    rows.append({
        "Dataset": f"{p['species'].replace(' ', '_')}_{p['antibiotic']}",
        "Species": p["species"],
        "Drug": p["antibiotic"],
        "Model": "N/A",
        "AUROC": "N/A",
        "F1_macro": "N/A",
        "N_samples": p.get("n_samples", 0),
        "N_test": "N/A",
        "Note": "Labels only — no genome features available locally",
    })

# ── Save multi-dataset results CSV ──────────────────────────────────────────
csv_path = os.path.join(EVALUATE_DIR, "multi_dataset_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "Dataset", "Species", "Drug", "Model", "AUROC", "F1_macro",
        "N_samples", "N_test", "Note"
    ])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

# ── Save ablation summary CSV ───────────────────────────────────────────────
ablation_csv_path = os.path.join(EVALUATE_DIR, "ablation_results.csv")
# Already saved by run_ablation.py — skip overwrite

# ── Combined JSON ───────────────────────────────────────────────────────────
combined = {
    "multi_dataset": rows,
    "ablation": ablation_results,
    "mendeley_limitation": mendeley_report["limitation"],
}
json_path = os.path.join(EVALUATE_DIR, "multi_dataset_results.json")
with open(json_path, "w") as f:
    json.dump(combined, f, indent=2)

print("Multi-Dataset Results")
print("=" * 90)
print(f"{'Dataset':<35s} {'AUROC':>8s} {'F1':>8s} {'N':>8s} {'Note'}")
print("-" * 90)
for row in rows:
    auroc = f"{row['AUROC']:.4f}" if isinstance(row['AUROC'], float) else row['AUROC']
    f1 = f"{row['F1_macro']:.4f}" if isinstance(row['F1_macro'], float) else row['F1_macro']
    print(f"{row['Dataset']:<35s} {auroc:>8s} {f1:>8s} {row['N_samples']:>8} {row['Note']}")

print(f"\nAblation Study (INH, 10 epochs each)")
print("=" * 60)
print(f"{'Config':<25s} {'AUROC':>8s} {'F1':>8s} {'Delta':>8s}")
print("-" * 60)
full_auroc = None
for r in ablation_results:
    if r["config"] == "full_kg_amr":
        full_auroc = r["auroc"]
for r in ablation_results:
    delta = r["auroc"] - full_auroc if full_auroc else 0
    print(f"{r['config']:<25s} {r['auroc']:8.4f} {r['f1_macro']:8.4f} {delta:+8.4f}")

print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
