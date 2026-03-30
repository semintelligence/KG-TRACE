"""
Step 1C: Fixed Pathway Extraction — node naming verified, efficient computation.
All values from actual NetworkX queries, no hardcoding.
"""
import sys, os, json, time, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "kg-amr-v2" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import networkx as nx

EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
KG_DIR = os.path.join(PROJECT_DIR, "kg")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[1/4] Loading KG graph and test data...")
G = nx.read_graphml(os.path.join(KG_DIR, "amr_graph.graphml"))
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"  Directed: {G.is_directed()}")

test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
gene_names = list(test_data["gene_names"])
attn_weights = test_data["attn_weights"]
labels = test_data["labels"]
preds = test_data["preds"]

DRUG = "INH"
assert DRUG in G.nodes(), f"Drug node '{DRUG}' not in graph"

# ── 2. Pre-compute gene→drug reachability ────────────────────────────────────
print(f"\n[2/4] Pre-computing gene→INH reachability...")

# For efficiency: pre-compute which genes have paths to INH
# and cache the first 3 unique paths per gene
gene_path_cache = {}
inh_genes = []  # genes with at least 1 path to INH

for gene in gene_names:
    if gene not in G.nodes():
        gene_path_cache[gene] = {"status": "gene_not_in_graph", "paths": []}
        continue
    
    paths = list(nx.all_simple_paths(G, source=gene, target=DRUG, cutoff=3))
    if paths:
        # Keep only paths with 'confers_resistance_to' edges (skip susceptibility)
        resistant_paths = []
        for p in paths:
            # Check each edge in the path
            has_resistance = False
            annotated_edges = []
            for k in range(len(p) - 1):
                edge_data = G.get_edge_data(p[k], p[k+1])
                rel = edge_data.get("relation", "unknown") if edge_data else "unknown"
                annotated_edges.append({"from": p[k], "to": p[k+1], "relation": rel})
                if rel == "confers_resistance_to":
                    has_resistance = True
            
            if has_resistance:
                resistant_paths.append({
                    "path_str": " → ".join(p),
                    "edges": annotated_edges,
                    "length": len(p) - 1,
                })
        
        if resistant_paths:
            inh_genes.append(gene)
        
        # Also store uncertain/susceptibility paths separately
        all_path_info = []
        for p in paths[:5]:  # limit to 5 per gene
            annotated_edges = []
            for k in range(len(p) - 1):
                edge_data = G.get_edge_data(p[k], p[k+1])
                rel = edge_data.get("relation", "unknown") if edge_data else "unknown"
                annotated_edges.append({"from": p[k], "to": p[k+1], "relation": rel})
            all_path_info.append({
                "path_str": " → ".join(p),
                "edges": annotated_edges,
                "length": len(p) - 1,
            })
        
        gene_path_cache[gene] = {
            "status": "paths_found",
            "n_total_paths": len(paths),
            "n_resistance_paths": len(resistant_paths),
            "sample_paths": all_path_info[:3],
            "sample_resistance_paths": resistant_paths[:3],
        }
    else:
        gene_path_cache[gene] = {"status": "no_path_to_INH", "paths": []}

print(f"  Genes with path to INH: {len(inh_genes)}/{len(gene_names)}")
print(f"  INH-linked genes: {inh_genes}")
for gene in gene_names:
    info = gene_path_cache[gene]
    status = info["status"]
    if status == "paths_found":
        print(f"    {gene:12s}: {info['n_total_paths']} total, {info['n_resistance_paths']} resistance")
    else:
        print(f"    {gene:12s}: {status}")

# ── 3. Extract per-genome pathways ───────────────────────────────────────────
print(f"\n[3/4] Extracting pathways for resistant test genomes...")
t0 = time.time()

resistant_mask = labels == 1
resistant_ids = test_ids[resistant_mask]
resistant_attn = attn_weights[resistant_mask]
n_resistant = len(resistant_ids)
print(f"  Resistant genomes: {n_resistant}")

gene_presence = test_data["gene_presence"]
resistant_presence = gene_presence[resistant_mask]

pathway_results = {}
genomes_with_path = 0
genomes_with_resistance_path = 0

for i in range(n_resistant):
    genome_id = str(resistant_ids[i])
    genome_attn = resistant_attn[i]
    genome_pres = resistant_presence[i]
    
    # Top-3 genes by attention weight
    top3_idx = np.argsort(genome_attn)[::-1][:3]
    
    genome_pathways = {}
    has_any_path = False
    has_resistance_path = False
    
    for gi in top3_idx:
        gene = gene_names[gi]
        weight = float(genome_attn[gi])
        present = bool(genome_pres[gi] > 0)
        
        cache = gene_path_cache[gene]
        
        genome_pathways[gene] = {
            "attention_weight": weight,
            "gene_present": present,
            "pathway_status": cache["status"],
            "n_paths": cache.get("n_total_paths", 0),
            "n_resistance_paths": cache.get("n_resistance_paths", 0),
            "sample_paths": cache.get("sample_paths", [])[:2],
        }
        
        if cache["status"] == "paths_found":
            has_any_path = True
            if cache.get("n_resistance_paths", 0) > 0:
                has_resistance_path = True
    
    if has_any_path:
        genomes_with_path += 1
    if has_resistance_path:
        genomes_with_resistance_path += 1
    
    pathway_results[genome_id] = {
        "true_label": "RESISTANT",
        "top3_genes": [(gene_names[gi], float(genome_attn[gi])) for gi in top3_idx],
        "pathways": genome_pathways,
        "has_valid_path": has_any_path,
        "has_resistance_path": has_resistance_path,
    }

elapsed = time.time() - t0
print(f"  Processed {n_resistant} genomes in {elapsed:.1f}s")

# ── 4. Report coverage ──────────────────────────────────────────────────────
print(f"\n[4/4] Pathway coverage report:")
coverage = genomes_with_path / n_resistant if n_resistant > 0 else 0.0
resistance_coverage = genomes_with_resistance_path / n_resistant if n_resistant > 0 else 0.0

print(f"  Resistant genomes with >= 1 path (any): {genomes_with_path}/{n_resistant} ({100*coverage:.1f}%)")
print(f"  Resistant genomes with >= 1 resistance path: {genomes_with_resistance_path}/{n_resistant} ({100*resistance_coverage:.1f}%)")

# Count which genes provide the paths
gene_contribution = {}
for gid, data in pathway_results.items():
    for gene, pdata in data["pathways"].items():
        if pdata["pathway_status"] == "paths_found":
            gene_contribution[gene] = gene_contribution.get(gene, 0) + 1

print(f"\n  Gene contributions to pathway coverage:")
for gene, count in sorted(gene_contribution.items(), key=lambda x: -x[1]):
    print(f"    {gene:12s}: appears in top-3 attention for {count} resistant genomes with paths")

# ── Save ────────────────────────────────────────────────────────────────────
output = {
    "drug": DRUG,
    "n_resistant_test": n_resistant,
    "n_with_any_path": genomes_with_path,
    "n_with_resistance_path": genomes_with_resistance_path,
    "pathway_coverage_any_pct": round(100 * coverage, 2),
    "pathway_coverage_resistance_pct": round(100 * resistance_coverage, 2),
    "gene_path_cache": gene_path_cache,
    "gene_contributions": gene_contribution,
    "per_genome_pathways": pathway_results,
}

out_path = os.path.join(EXPLAIN_DIR, "pathway_explanations.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Saved to {out_path}")
print(f"  File size: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")
print("DONE — pathway_explain_fixed.py")
