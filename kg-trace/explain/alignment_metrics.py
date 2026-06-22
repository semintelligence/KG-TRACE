"""
Step 8D: BCS + Spearman rho — Alignment Metrics
All values computed honestly from actual model outputs.
"""
import sys, os, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
from scipy.stats import spearmanr

EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/3] Loading attention weights and SHAP values...")

# Load test outputs
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
gene_names = list(test_data["gene_names"])
attn_weights = test_data["attn_weights"]  # [n_test, 26]
labels = test_data["labels"]
preds = test_data["preds"]

# Load SHAP values
shap_data = np.load(os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"), allow_pickle=True)
shap_values = shap_data["shap_values"]
feature_names = list(shap_data["feature_names"])

# Load CARD gene set
with open(os.path.join(PROJECT_DIR, "kg/gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
card_gene_set = set(gene_mechanism.keys())

n_test = len(test_ids)
print(f"  Test samples: {n_test}")
print(f"  Genes: {len(gene_names)}")
print(f"  SHAP features: {shap_values.shape[1]}")

# ── 2. BCS Computation ──────────────────────────────────────────────────────
print("\n[2/3] Computing BCS (Biological Consistency Score)...")

def compute_bcs(top_shap_features, card_gene_set, N=10, drug="RIF"):
    """BCS = fraction of top-N SHAP features that are biologically grounded in the catalogue."""
    mapped = []
    for f in top_shap_features[:N]:
        gene = f.split(":")[0]
        if gene in card_gene_set or f in card_gene_set:
            mapped.append(f)
    return len(mapped) / N

# Global BCS: top-N features by mean |SHAP| across test set
shap_mean_abs = np.abs(shap_values).mean(axis=0)
global_top_idx = np.argsort(shap_mean_abs)[::-1]
global_top_features = [feature_names[i] for i in global_top_idx]

for N in [5, 10, 20, 50]:
    bcs = compute_bcs(global_top_features, card_gene_set, N=N)
    top_genes = [feature_names[global_top_idx[i]].split(":")[0] for i in range(N)]
    mapped = [g for g in top_genes if g in card_gene_set]
    print(f"  BCS@{N}: {bcs:.2f} ({len(mapped)}/{N} mapped)")
    print(f"    Top-{N} genes: {top_genes}")
    print(f"    Mapped: {mapped}")

# ── Baseline BCS (Genomic-Only) ─────────────────────────────────────────────
print("\n[2.5/3] Computing BCS for Genomic-Only baseline...")
genomic_shap_path = os.path.join(EXPLAIN_DIR, "shap_raw_genomic_only.npz")
genomic_bcs_10 = 0.0
genomic_bcs_20 = 0.0
genomic_bcs_50 = 0.0

if os.path.exists(genomic_shap_path):
    g_shap_data = np.load(genomic_shap_path, allow_pickle=True)
    g_shap_values = g_shap_data["shap_values"]
    g_feature_names = list(g_shap_data["feature_names"])
    
    g_shap_mean_abs = np.abs(g_shap_values).mean(axis=0)
    g_global_top_idx = np.argsort(g_shap_mean_abs)[::-1]
    g_global_top_features = [g_feature_names[i] for i in g_global_top_idx]
    
    genomic_bcs_10 = float(compute_bcs(g_global_top_features, card_gene_set, N=10))
    genomic_bcs_20 = float(compute_bcs(g_global_top_features, card_gene_set, N=20))
    genomic_bcs_50 = float(compute_bcs(g_global_top_features, card_gene_set, N=50))
    
    print(f"  Genomic-Only BCS@10: {genomic_bcs_10:.2f}")
    print(f"  Genomic-Only BCS@20: {genomic_bcs_20:.2f}")
    print(f"  Genomic-Only BCS@50: {genomic_bcs_50:.2f}")
else:
    print("  WARNING: shap_raw_genomic_only.npz not found.")

# Per-genome BCS
per_genome_bcs = []
n_test = shap_values.shape[0]
for i in range(n_test):
    genome_shap = np.abs(shap_values[i])
    genome_top = np.argsort(genome_shap)[::-1][:10]
    top_feats = [feature_names[fi] for fi in genome_top]
    bcs = compute_bcs(top_feats, card_gene_set, N=10)
    per_genome_bcs.append(bcs)

per_genome_bcs = np.array(per_genome_bcs)
print(f"\n  Per-genome BCS@10 stats:")
print(f"    Mean: {per_genome_bcs.mean():.4f}")
print(f"    Std:  {per_genome_bcs.std():.4f}")
print(f"    Min:  {per_genome_bcs.min():.4f}")
print(f"    Max:  {per_genome_bcs.max():.4f}")

# ── 3. Spearman rho ─────────────────────────────────────────────────────────
print("\n[3/3] Computing Spearman rho...")

# Compare model attention-based gene importance with SHAP-derived gene importance.
# Model importance = mean attention weight per gene across test set
# SHAP importance = mean |SHAP| aggregated per gene

# Gene-level attention importance
attn_importance = attn_weights.mean(axis=0)  # [26]
attn_ranks = np.argsort(np.argsort(-attn_importance)) + 1  # rank (1 = most important)

# Gene-level SHAP importance (aggregate SHAP per gene)
gene_shap_importance = np.zeros(len(gene_names))
for fi, feat in enumerate(feature_names):
    gene = feat.split(":")[0]
    if gene in gene_names:
        gi = gene_names.index(gene)
        gene_shap_importance[gi] += shap_mean_abs[fi]

shap_ranks = np.argsort(np.argsort(-gene_shap_importance)) + 1

# Spearman correlation
rho, pvalue = spearmanr(attn_ranks, shap_ranks)
print(f"  Attention ranks: {list(zip(gene_names[:5], attn_ranks[:5]))}")
print(f"  SHAP ranks:      {list(zip(gene_names[:5], shap_ranks[:5]))}")
print(f"  Spearman rho = {rho:.4f}")
print(f"  p-value      = {pvalue:.6f}")

if pvalue > 0.05:
    print("  WARNING: Correlation NOT statistically significant (p > 0.05)")
else:
    print(f"  Correlation IS statistically significant (p = {pvalue:.6f})")

# Also compute Spearman between CARD-known resistance relevance and model importance
# Use the KG relation "confers_resistance_to" as a proxy for CARD relevance
# Genes with more R mutations in the KG are more "relevant"
import networkx as nx
G = nx.read_graphml(os.path.join(PROJECT_DIR, "kg/amr_graph.graphml"))

# Count resistance edges per gene
gene_resistance_count = {}
for gene in gene_names:
    # Count mutations of this gene that confer resistance
    count = 0
    for nbr in G.neighbors(gene):
        # gene -> mutation
        for nbr2 in G.neighbors(nbr):
            edge_data = G.get_edge_data(nbr, nbr2)
            if edge_data and edge_data.get("relation") == "confers_resistance_to":
                count += 1
    gene_resistance_count[gene] = count

card_relevance = np.array([gene_resistance_count.get(g, 0) for g in gene_names], dtype=float)
card_ranks = np.argsort(np.argsort(-card_relevance)) + 1

rho_card, pvalue_card = spearmanr(attn_ranks, card_ranks)
print(f"\n  Attention vs CARD resistance edges:")
print(f"    Spearman rho = {rho_card:.4f}")
print(f"    p-value      = {pvalue_card:.6f}")
if pvalue_card > 0.05:
    print("    WARNING: Correlation NOT statistically significant (p > 0.05)")

rho_shap_card, pvalue_shap_card = spearmanr(shap_ranks, card_ranks)
print(f"\n  SHAP vs CARD resistance edges:")
print(f"    Spearman rho = {rho_shap_card:.4f}")
print(f"    p-value      = {pvalue_shap_card:.6f}")
if pvalue_shap_card > 0.05:
    print("    WARNING: Correlation NOT statistically significant (p > 0.05)")

# ── Save alignment metrics ───────────────────────────────────────────────────
metrics = {
    "bcs_global_5": float(compute_bcs(global_top_features, card_gene_set, N=5)),
    "bcs_global_10": float(compute_bcs(global_top_features, card_gene_set, N=10)),
    "bcs_global_20": float(compute_bcs(global_top_features, card_gene_set, N=20)),
    "bcs_global_50": float(compute_bcs(global_top_features, card_gene_set, N=50)),
    "bcs_genomic_only_10": genomic_bcs_10,
    "bcs_genomic_only_20": genomic_bcs_20,
    "bcs_genomic_only_50": genomic_bcs_50,
    "bcs_per_genome_mean": float(per_genome_bcs.mean()),
    "bcs_per_genome_std": float(per_genome_bcs.std()),
    "spearman_attn_vs_shap": {"rho": float(rho), "pvalue": float(pvalue)},
    "spearman_attn_vs_card": {"rho": float(rho_card), "pvalue": float(pvalue_card)},
    "spearman_shap_vs_card": {"rho": float(rho_shap_card), "pvalue": float(pvalue_shap_card)},
    "gene_importance_table": [
        {
            "gene": gene_names[i],
            "attn_mean": float(attn_importance[i]),
            "attn_rank": int(attn_ranks[i]),
            "shap_importance": float(gene_shap_importance[i]),
            "shap_rank": int(shap_ranks[i]),
            "card_resistance_edges": int(card_relevance[i]),
            "card_rank": int(card_ranks[i]),
        }
        for i in range(len(gene_names))
    ],
    "per_genome_bcs": per_genome_bcs.tolist(),
}

with open(os.path.join(EXPLAIN_DIR, "alignment_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n  Saved alignment metrics to {EXPLAIN_DIR}/alignment_metrics.json")
print("DONE — alignment_metrics.py")
