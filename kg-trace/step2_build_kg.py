"""
Step 2: Build AMR Knowledge Graph from CRyPTIC EFFECTS data.
Nodes: genes, mutations, drugs, mechanisms (inferred from gene function)
Edges: gene→has_mutation→mutation, mutation→confers_resistance→drug,
       gene→belongs_to→mechanism
Output: triples TSV for PyKEEN + NetworkX graph for path-based explainability.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from paths import *

import pandas as pd
import numpy as np
import networkx as nx
import json

KG_DIR = os.path.join(PROJECT_DIR, "kg")
os.makedirs(KG_DIR, exist_ok=True)

t0 = time.time()

# ── 1. Load EFFECTS.parquet ─────────────────────────────────────────────────
print("[1/4] Loading EFFECTS.parquet...")
effects = pd.read_parquet(os.path.join(ZENODO_DIR, "EFFECTS.parquet"))
print(f"  Shape: {effects.shape}")

# Extract unique (GENE, MUTATION, DRUG, PREDICTION) tuples
idx = effects.index
genes = idx.get_level_values("GENE")
mutations = idx.get_level_values("MUTATION")
drugs = idx.get_level_values("DRUG")
predictions = effects["PREDICTION"].values

# Filter to non-null genes
valid = pd.notna(genes) & pd.notna(mutations) & pd.notna(drugs)
genes = genes[valid]
mutations = mutations[valid]
drugs = drugs[valid]
predictions = predictions[valid]

# Create a DataFrame of unique relationships
rel_df = pd.DataFrame({
    "gene": genes.astype(str),
    "mutation": mutations.astype(str),
    "drug": drugs.astype(str),
    "prediction": predictions.astype(str)
}).drop_duplicates()
print(f"  Unique (gene, mutation, drug, prediction) tuples: {len(rel_df):,}")

# ── 2. Define gene → mechanism mapping ──────────────────────────────────────
print("\n[2/4] Building gene → mechanism mapping...")

# Based on WHO TB mutation catalogue: map each gene to its biological mechanism
GENE_MECHANISM = {
    # Cell wall synthesis
    "embB": "cell_wall_synthesis",
    "inhA": "cell_wall_synthesis",
    "fabG1": "cell_wall_synthesis",
    "katG": "cell_wall_synthesis",
    
    # DNA replication
    "gyrA": "dna_replication",
    "gyrB": "dna_replication",
    
    # RNA transcription
    "rpoB": "rna_transcription",
    
    # Protein synthesis
    "rpsL": "protein_synthesis",
    "rplC": "protein_synthesis",
    "rrl": "protein_synthesis",
    "rrs": "protein_synthesis",
    "gid": "protein_synthesis",
    "tlyA": "protein_synthesis",
    "eis": "protein_synthesis",
    
    # Energy metabolism
    "atpE": "energy_metabolism",
    "fbiA": "energy_metabolism",
    "fbiB": "energy_metabolism",
    "fbiC": "energy_metabolism",
    "fgd1": "energy_metabolism",
    "ddn": "energy_metabolism",
    
    # Nicotinamide metabolism
    "pncA": "nicotinamide_metabolism",
    
    # Drug efflux
    "mmpL5": "drug_efflux",
    "Rv0678": "drug_efflux",
    
    # Thioamide activation
    "ethA": "thioamide_activation",
    
    # Regulatory
    "Rv2983": "regulatory",
    "pepQ": "regulatory",
}

print(f"  Mapped {len(GENE_MECHANISM)} genes to mechanisms")
unmapped = set(rel_df["gene"].unique()) - set(GENE_MECHANISM.keys())
if unmapped:
    print(f"  WARNING: Unmapped genes: {unmapped}")
    # Assign 'unknown_mechanism' for any unmapped
    for g in unmapped:
        GENE_MECHANISM[g] = "unknown_mechanism"

# ── 3. Build KG triples ────────────────────────────────────────────────────
print("\n[3/4] Generating KG triples...")

triples = []

# 3a. Gene → has_mutation → Mutation (unique per gene)
gene_mutations = rel_df[["gene", "mutation"]].drop_duplicates()
for _, row in gene_mutations.iterrows():
    mut_node = f"{row['gene']}:{row['mutation']}"
    triples.append((row["gene"], "has_mutation", mut_node))

# 3b. Mutation → confers_resistance_to / confers_susceptibility_to → Drug
for _, row in rel_df.iterrows():
    mut_node = f"{row['gene']}:{row['mutation']}"
    if row["prediction"] == "R":
        triples.append((mut_node, "confers_resistance_to", row["drug"]))
    elif row["prediction"] == "S":
        triples.append((mut_node, "confers_susceptibility_to", row["drug"]))
    else:  # U = unknown
        triples.append((mut_node, "has_uncertain_effect_on", row["drug"]))

# 3c. Gene → belongs_to → Mechanism
for gene, mechanism in GENE_MECHANISM.items():
    triples.append((gene, "belongs_to", mechanism))

# 3d. Drug → targets → Mechanism (inferred from gene-drug-mechanism path)
drug_mechs = set()
for _, row in rel_df[rel_df["prediction"] == "R"].iterrows():
    mech = GENE_MECHANISM.get(row["gene"])
    if mech:
        drug_mechs.add((row["drug"], mech))
for drug, mech in drug_mechs:
    triples.append((drug, "targets", mech))

# Deduplicate
triples = list(set(triples))
print(f"  Total unique triples: {len(triples):,}")

# Count by relation type
from collections import Counter
rel_counts = Counter(t[1] for t in triples)
for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
    print(f"    {rel}: {count:,}")

# ── 4. Save outputs ────────────────────────────────────────────────────────
print("\n[4/4] Saving outputs...")

# 4a. TSV for PyKEEN
triples_df = pd.DataFrame(triples, columns=["head", "relation", "tail"])
triples_path = os.path.join(KG_DIR, "amr_triples.tsv")
triples_df.to_csv(triples_path, sep="\t", index=False, header=False)
print(f"  Saved {len(triples_df):,} triples to amr_triples.tsv")

# 4b. NetworkX graph for path-based explainability
G = nx.DiGraph()
for h, r, t in triples:
    G.add_edge(h, t, relation=r)

nx.write_graphml(G, os.path.join(KG_DIR, "amr_graph.graphml"))
print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 4c. Entity and relation mappings
entities = sorted(set(triples_df["head"]) | set(triples_df["tail"]))
relations = sorted(set(triples_df["relation"]))
entity_to_id = {e: i for i, e in enumerate(entities)}
relation_to_id = {r: i for i, r in enumerate(relations)}

with open(os.path.join(KG_DIR, "entity_to_id.json"), "w") as f:
    json.dump(entity_to_id, f)
with open(os.path.join(KG_DIR, "relation_to_id.json"), "w") as f:
    json.dump(relation_to_id, f)
with open(os.path.join(KG_DIR, "gene_mechanism.json"), "w") as f:
    json.dump(GENE_MECHANISM, f, indent=2)

print(f"  Entities: {len(entities)}, Relations: {len(relations)}")
print(f"  Entity types: {len([e for e in entities if ':' not in e and e not in GENE_MECHANISM.values() and len(e) <= 4])} drugs, "
      f"{len(GENE_MECHANISM)} genes, "
      f"{len([e for e in entities if ':' in e])} mutations, "
      f"{len(set(GENE_MECHANISM.values()))} mechanisms")

# 4d. Summary
summary = {
    "n_triples": len(triples),
    "n_entities": len(entities),
    "n_relations": len(relations),
    "relation_counts": dict(rel_counts),
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "elapsed_seconds": round(time.time() - t0, 1)
}
with open(os.path.join(KG_DIR, "kg_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Elapsed: {time.time()-t0:.1f}s")
print(f"  All outputs in: {KG_DIR}")
print("\nDONE")
