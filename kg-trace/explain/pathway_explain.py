"""
Step 8E: Pathway Reasoning — Extract KG paths from top attention genes to drugs.
Queries actual NetworkX graph. Reports "No path found" if none exists.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import networkx as nx

EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
KG_DIR = os.path.join(PROJECT_DIR, "kg")

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("[1/3] Loading KG graph and test data...")

G = nx.read_graphml(os.path.join(KG_DIR, "amr_graph.graphml"))
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
gene_names = list(test_data["gene_names"])
attn_weights = test_data["attn_weights"]
labels = test_data["labels"]
preds = test_data["preds"]

# Drug target: INH (isoniazid)
DRUG = "INH"

# ── 2. Pathway extraction function ──────────────────────────────────────────

def extract_pathway(G, gene_node, antibiotic_node, cutoff=3):
    """Find all simple paths from gene to antibiotic in the KG."""
    try:
        paths = list(nx.all_simple_paths(
            G, source=gene_node, target=antibiotic_node, cutoff=cutoff
        ))
        if paths:
            # Annotate each edge with its relation
            annotated = []
            for path in paths:
                edges = []
                for k in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[k], path[k + 1])
                    rel = edge_data.get("relation", "unknown") if edge_data else "unknown"
                    edges.append({"from": path[k], "to": path[k + 1], "relation": rel})
                annotated.append({
                    "nodes": path,
                    "edges": edges,
                    "length": len(path) - 1,
                })
            return annotated
        else:
            return "No path found in CARD KG"
    except nx.NodeNotFound as e:
        return f"Node not found: {e}"


# ── 3. Run pathway extraction for resistant test genomes ─────────────────────
print(f"\n[2/3] Extracting pathways for resistant test genomes (drug={DRUG})...")

resistant_mask = labels == 1
resistant_ids = test_ids[resistant_mask]
resistant_attn = attn_weights[resistant_mask]
n_resistant = len(resistant_ids)
print(f"  Resistant genomes in test set: {n_resistant}")

# Check if drug node exists in graph
if DRUG not in G.nodes():
    print(f"  WARNING: Drug '{DRUG}' not found in graph. Available drug nodes:")
    possible_drugs = [n for n in G.nodes() if len(n) <= 4 and n.isupper()]
    print(f"    {possible_drugs}")

pathway_results = {}
genomes_with_path = 0

for i in range(n_resistant):
    genome_id = str(resistant_ids[i])
    genome_attn = resistant_attn[i]

    # Top-3 genes by attention weight
    top3_idx = np.argsort(genome_attn)[::-1][:3]
    top3_genes = [(gene_names[gi], float(genome_attn[gi])) for gi in top3_idx]

    genome_paths = {}
    has_any_path = False

    for gene, weight in top3_genes:
        result = extract_pathway(G, gene, DRUG, cutoff=3)
        genome_paths[gene] = {
            "attention_weight": weight,
            "pathways": result,
        }
        if isinstance(result, list) and len(result) > 0:
            has_any_path = True

    if has_any_path:
        genomes_with_path += 1

    pathway_results[genome_id] = {
        "true_label": "RESISTANT",
        "top3_genes": top3_genes,
        "pathways": genome_paths,
        "has_valid_path": has_any_path,
    }

# ── Report coverage ──────────────────────────────────────────────────────────
print(f"\n[3/3] Pathway coverage report:")
coverage = genomes_with_path / n_resistant if n_resistant > 0 else 0.0
print(f"  Resistant genomes with >= 1 valid path: {genomes_with_path}/{n_resistant} ({100*coverage:.1f}%)")

# Count unique pathways
all_paths = []
gene_path_counts = {}
for gid, data in pathway_results.items():
    for gene, pdata in data["pathways"].items():
        if isinstance(pdata["pathways"], list):
            for p in pdata["pathways"]:
                path_str = " -> ".join(p["nodes"])
                all_paths.append(path_str)
                gene_path_counts[gene] = gene_path_counts.get(gene, 0) + 1

from collections import Counter
path_counter = Counter(all_paths)
print(f"\n  Total path instances: {len(all_paths)}")
print(f"  Unique paths: {len(path_counter)}")

print(f"\n  Top-10 most common pathways:")
for rank, (path, count) in enumerate(path_counter.most_common(10)):
    print(f"    {rank+1:2d}. ({count:4d}x) {path}")

print(f"\n  Genes with pathways to {DRUG}:")
for gene, count in sorted(gene_path_counts.items(), key=lambda x: -x[1]):
    print(f"    {gene:12s}: {count:4d} path instances")

# ── Save pathway explanations ────────────────────────────────────────────────
output = {
    "drug": DRUG,
    "n_resistant_test": n_resistant,
    "n_with_valid_path": genomes_with_path,
    "pathway_coverage_pct": round(100 * coverage, 2),
    "top_10_paths": path_counter.most_common(10),
    "gene_path_counts": gene_path_counts,
    "per_genome_pathways": pathway_results,
}

with open(os.path.join(EXPLAIN_DIR, "pathway_explanations.json"), "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n  Saved to {EXPLAIN_DIR}/pathway_explanations.json")
print("DONE — pathway_explain.py")
