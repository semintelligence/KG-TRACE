"""
Step 8F: Generate per-genome JSON reports for all test genomes.
All values from actual model outputs, nothing hardcoded.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np

EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
REPORTS_DIR = os.path.join(EXPLAIN_DIR, "per_genome_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── 1. Load all data ────────────────────────────────────────────────────────
print("[1/3] Loading data...")

# Test outputs
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = list(test_data["test_ids"])
gene_names = list(test_data["gene_names"])
attn_weights = test_data["attn_weights"]    # [n_test, 26]
gate_values = test_data["gate_values"]      # [n_test, 128]
preds = test_data["preds"]
probs = test_data["probs"]
labels = test_data["labels"]
gene_presence = test_data["gene_presence"]  # [n_test, 26]

# SHAP values
shap_data = np.load(os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"), allow_pickle=True)
shap_values = shap_data["shap_values"]
feature_names = list(shap_data["feature_names"])

# Alignment metrics
with open(os.path.join(EXPLAIN_DIR, "alignment_metrics.json")) as f:
    alignment = json.load(f)

# CARD gene set
with open(os.path.join(PROJECT_DIR, "kg/gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
card_gene_set = set(gene_mechanism.keys())

# Pathway explanations (may be large)
pathway_path = os.path.join(EXPLAIN_DIR, "pathway_explanations.json")
pathways = {}
if os.path.exists(pathway_path):
    try:
        with open(pathway_path) as f:
            pathways = json.load(f)
        print(f"  Loaded pathway explanations for {len(pathways)} genomes")
    except Exception as e:
        print(f"  WARNING: Could not load pathway_explanations.json: {e}")

n_test = len(test_ids)
print(f"  Test samples: {n_test}")

# ── 2. Global metrics ───────────────────────────────────────────────────────
print("\n[2/3] Computing global stats...")

gate_means = gate_values.mean(axis=1)
global_bcs = alignment.get("bcs_global_10", None)
spearman_attn_shap = alignment.get("spearman_attn_vs_shap", {})

# ── 3. Generate reports ─────────────────────────────────────────────────────
print("\n[3/3] Generating per-genome JSON reports...")

for i in range(n_test):
    genome_id = str(test_ids[i])
    
    # True label and prediction
    true_label = "RESISTANT" if labels[i] == 1 else "SUSCEPTIBLE"
    prediction = "RESISTANT" if preds[i] == 1 else "SUSCEPTIBLE"
    correct = bool(labels[i] == preds[i])
    confidence = float(probs[i]) if preds[i] == 1 else float(1.0 - probs[i])
    
    # Top genes by attention
    attn_sorted = np.argsort(attn_weights[i])[::-1]
    top_genes = []
    for idx in attn_sorted[:5]:
        top_genes.append({
            "gene": gene_names[idx],
            "weight": float(attn_weights[i, idx]),
            "present": bool(gene_presence[i, idx] > 0),
        })
    
    # Gate
    gate_mean = float(gate_means[i])
    fusion_mode = "KG-dominated" if gate_mean < 0.5 else "genomic-dominated"
    
    # SHAP top k-mers
    genome_shap = np.abs(shap_values[i])
    genome_top_idx = np.argsort(genome_shap)[::-1][:10]
    shap_top_kmers = []
    shap_card_mapped = []
    unmapped_count = 0
    for fi in genome_top_idx:
        feat = feature_names[fi]
        gene = feat.split(":")[0]
        shap_top_kmers.append(feat)
        if gene in card_gene_set:
            shap_card_mapped.append(gene)
        else:
            unmapped_count += 1
    
    bcs = len(shap_card_mapped) / 10
    
    # Pathway explanation
    pw = pathways.get(genome_id, None)
    if pw is not None:
        # Summarize pathway — may contain multiple paths
        if isinstance(pw, list) and len(pw) > 0:
            pathway_explanation = f"{len(pw)} path(s) found"
        elif isinstance(pw, dict):
            pathway_explanation = pw.get("summary", "See pathway_explanations.json")
        elif isinstance(pw, str):
            pathway_explanation = pw
        else:
            pathway_explanation = "No path found"
    else:
        pathway_explanation = "Not computed"
    
    report = {
        "genome_id": genome_id,
        "true_label": true_label,
        "prediction": prediction,
        "correct": correct,
        "confidence": round(confidence, 4),
        "top_genes_by_attention": top_genes,
        "gate_mean": round(gate_mean, 6),
        "fusion_mode": fusion_mode,
        "pathway_explanation": pathway_explanation,
        "shap_top_features": shap_top_kmers,
        "shap_card_mapped": list(set(shap_card_mapped)),
        "unmapped_shap_count": unmapped_count,
        "bcs": round(bcs, 2),
        "spearman_rho": spearman_attn_shap.get("rho", None),
    }
    
    # Save — use sanitized filename
    safe_id = genome_id.replace("/", "_").replace(" ", "_")
    report_path = os.path.join(REPORTS_DIR, f"{safe_id}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

print(f"  Generated {n_test} per-genome reports in {REPORTS_DIR}/")

# Also save a summary of all reports
summary = {
    "total_reports": n_test,
    "correct_predictions": int((labels == preds).sum()),
    "accuracy": float((labels == preds).mean()),
    "mean_confidence": float(np.mean([
        probs[i] if preds[i] == 1 else 1.0 - probs[i]
        for i in range(n_test)
    ])),
    "mean_gate": float(gate_means.mean()),
    "pct_kg_dominated": float(100 * (gate_means < 0.5).sum() / n_test),
    "global_bcs_10": global_bcs,
    "spearman_attn_vs_shap": spearman_attn_shap,
}
with open(os.path.join(REPORTS_DIR, "_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Summary saved to {REPORTS_DIR}/_summary.json")
print("DONE — generate_per_genome_reports.py")
