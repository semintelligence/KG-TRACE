"""
Stage M5: Build Per-Species Knowledge Graph + RotatE Embeddings
Creates a gene-drug-mechanism KG from BV-BRC AMR annotations and trains RotatE.

Node types:  gene node, drug node, mechanism class node
Edge types:  gene→confers_resistance_to→drug
             gene→belongs_to_class→mechanism_class

Outputs per species:
  kg/species/{species}/amr_triples.tsv          (head, relation, tail)
  kg/species/{species}/entity_to_id.json        (entity name → int ID)
  kg/species/{species}/relation_to_id.json      (relation type → int ID)
  kg/species/{species}/gene_mechanism.json      (gene → mechanism class)
  kg/species/{species}/entity_embeddings.npy    (RotatE embeddings, float32 magnitude)
  kg/species/{species}/kg_summary.json          (stats)
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR, KG_EMBED_DIM

ANNOT_DIR  = os.path.join(PROJECT_DIR, "features/annotations")
KG_BASE    = os.path.join(PROJECT_DIR, "kg/species")
os.makedirs(KG_BASE, exist_ok=True)

SPECIES_LIST = [
    "Ecoli_ampicillin",
    "Kpneumoniae_cipro",
    "Kpneumoniae_carbapenem",
    "Abaumannii_carbapenem",
]

# Drug labels per species (used as drug node name)
SPECIES_DRUG = {
    "Ecoli_ampicillin":        "ampicillin",
    "Kpneumoniae_cipro":       "ciprofloxacin",
    "Kpneumoniae_carbapenem":  "carbapenem",
    "Abaumannii_carbapenem":   "carbapenem",
}

# Known gene family → mechanism class mapping
# Covers major Gram-negative resistance gene families
GENE_FAMILY_MECHANISM = {
    # Beta-lactamases
    "bla": "beta_lactamase",
    "TEM": "beta_lactamase",
    "SHV": "beta_lactamase",
    "CTX": "beta_lactamase",
    "OXA": "beta_lactamase",
    "NDM": "carbapenemase",
    "KPC": "carbapenemase",
    "VIM": "carbapenemase",
    "IMP": "carbapenemase",
    "CMY": "beta_lactamase",
    "DHA": "beta_lactamase",
    "FOX": "beta_lactamase",
    "ACC": "beta_lactamase",
    # Aminoglycoside resistance
    "aac": "aminoglycoside_modifying_enzyme",
    "aph": "aminoglycoside_modifying_enzyme",
    "aad": "aminoglycoside_modifying_enzyme",
    "ant": "aminoglycoside_modifying_enzyme",
    # Fluoroquinolone resistance
    "qnr": "quinolone_protection",
    "gyrA": "dna_gyrase_mutation",
    "gyrB": "dna_gyrase_mutation",
    "parC": "topoisomerase_mutation",
    "parE": "topoisomerase_mutation",
    "oqxA": "efflux_pump",
    "oqxB": "efflux_pump",
    # Efflux pumps
    "tet": "efflux_pump",
    "mexA": "efflux_pump",
    "mexB": "efflux_pump",
    "mexC": "efflux_pump",
    "mexD": "efflux_pump",
    "acrA": "efflux_pump",
    "acrB": "efflux_pump",
    "tolC": "efflux_pump",
    "adeA": "efflux_pump",
    "adeB": "efflux_pump",
    "adeC": "efflux_pump",
    # Trimethoprim/sulfonamide
    "dfr": "dihydrofolate_reductase",
    "sul": "sulfonamide_resistance",
    # Chloramphenicol
    "cat": "chloramphenicol_acetyltransferase",
    "cml": "chloramphenicol_efflux",
    # Other
    "mcr": "colistin_resistance",
    "erm": "macrolide_resistance",
    "mph": "macrolide_resistance",
    "msr": "macrolide_efflux",
    "van": "glycopeptide_resistance",
}


def classify_gene(gene_name):
    """Return mechanism class for a gene based on known family patterns."""
    g_upper = gene_name.upper()
    for prefix, mech in GENE_FAMILY_MECHANISM.items():
        if g_upper.startswith(prefix.upper()):
            return mech
    return "other_mechanism"


# ── Per-species KG building ──────────────────────────────────────────────────
for species in SPECIES_LIST:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Building KG for: {species}")

    # Load gene annotations from Stage M3
    genes_file = os.path.join(ANNOT_DIR, f"{species}_genes.json")
    if not os.path.exists(genes_file):
        print(f"  [SKIP] No gene annotations — run stage_m3 first")
        continue
    with open(genes_file) as f:
        gene_to_col = json.load(f)
    genes_list = sorted(gene_to_col, key=gene_to_col.get)

    drug_name = SPECIES_DRUG[species]
    print(f"  Genes: {len(genes_list)}")
    print(f"  Drug: {drug_name}")

    # ── Build triples ──────────────────────────────────────────────────────
    triples = []
    gene_mech_map = {}
    unique_entities = set()
    unique_mech = set()

    for gene in genes_list:
        mech = classify_gene(gene)
        gene_mech_map[gene] = mech
        unique_entities.add(gene)
        unique_entities.add(drug_name)
        unique_mech.add(mech)
        # Triple 1: gene → confers_resistance_to → drug
        triples.append((gene, "confers_resistance_to", drug_name))
        # Triple 2: gene → belongs_to_class → mechanism_class
        triples.append((gene, "belongs_to_class", mech))
        unique_entities.add(mech)

    # Add mechanism → involves → drug links too (enriches KG structure)
    for mech in unique_mech:
        triples.append((mech, "involves_drug", drug_name))

    print(f"  Triples: {len(triples)}")
    print(f"  Entities: {len(unique_entities)}")
    print(f"  Mechanism classes: {len(unique_mech)}")

    # ── Entity and relation ID maps ─────────────────────────────────────────
    entities = sorted(unique_entities)
    relations = ["confers_resistance_to", "belongs_to_class", "involves_drug"]
    entity_to_id = {e: i for i, e in enumerate(entities)}
    relation_to_id = {r: i for i, r in enumerate(relations)}

    # ── Save triples TSV ───────────────────────────────────────────────────
    sp_kg_dir = os.path.join(KG_BASE, species)
    os.makedirs(sp_kg_dir, exist_ok=True)
    triples_path = os.path.join(sp_kg_dir, "amr_triples.tsv")
    with open(triples_path, "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")

    with open(os.path.join(sp_kg_dir, "entity_to_id.json"), "w") as f:
        json.dump(entity_to_id, f, indent=2)
    with open(os.path.join(sp_kg_dir, "relation_to_id.json"), "w") as f:
        json.dump(relation_to_id, f, indent=2)
    with open(os.path.join(sp_kg_dir, "gene_mechanism.json"), "w") as f:
        json.dump(gene_mech_map, f, indent=2)

    # ── Train RotatE with PyKEEN ───────────────────────────────────────────
    if len(genes_list) < 2:
        print(f"  [WARN] Only {len(genes_list)} genes — KG too small for RotatE. "
              f"Creating random init embeddings.")
        n_ent = len(entity_to_id)
        entity_emb = np.random.randn(n_ent, KG_EMBED_DIM).astype(np.float32) * 0.1
    else:
        print(f"  Training RotatE (dim={KG_EMBED_DIM})...")
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline as kg_pipeline

        # Build PyKEEN TriplesFactory
        heads      = [h for h, r, t in triples]
        relations_ = [r for h, r, t in triples]
        tails      = [t for h, r, t in triples]
        triples_arr = np.array([[h, r, t] for h, r, t in triples])

        tf = TriplesFactory.from_labeled_triples(
            triples=triples_arr,
            entity_to_id=None,
            relation_to_id=None,
        )

        # Determine epochs: more genes → more epochs
        n_epochs = min(500, max(100, len(triples) * 2))
        print(f"  PyKEEN TriplesFactory: {tf.num_entities} entities, "
              f"{tf.num_relations} relations, {tf.num_triples} triples")
        print(f"  Training {n_epochs} epochs...")

        result = kg_pipeline(
            training=tf,
            testing=tf,    # no separate test set for KG training
            model="RotatE",
            model_kwargs={"embedding_dim": KG_EMBED_DIM // 2},  # RotatE uses dim/2 complex
            training_loop="sLCWA",
            training_kwargs={"num_epochs": n_epochs, "batch_size": min(256, len(triples))},
            optimizer="adam",
            optimizer_kwargs={"lr": 0.01},
            negative_sampler="basic",
            random_seed=42,
            device="cpu",    # avoid MPS for small training
        )

        # Extract entity embeddings
        model = result.model
        # entity_representations[0] contains the RotatE complex embeddings
        emb_raw = model.entity_representations[0]().detach().cpu().numpy()
        if np.iscomplexobj(emb_raw):
            entity_emb = np.abs(emb_raw).astype(np.float32)
        else:
            entity_emb = emb_raw.astype(np.float32)

        # Pad/project to KG_EMBED_DIM if necessary
        if entity_emb.shape[1] != KG_EMBED_DIM:
            # zero-pad or truncate
            padded = np.zeros((entity_emb.shape[0], KG_EMBED_DIM), dtype=np.float32)
            copy_dim = min(entity_emb.shape[1], KG_EMBED_DIM)
            padded[:, :copy_dim] = entity_emb[:, :copy_dim]
            entity_emb = padded

        # Re-order to match our entity_to_id (PyKEEN may have different ordering)
        pykeen_e2id = tf.entity_to_id
        final_emb = np.zeros((len(entity_to_id), KG_EMBED_DIM), dtype=np.float32)
        for ent, our_id in entity_to_id.items():
            pykeen_id = pykeen_e2id.get(ent)
            if pykeen_id is not None:
                final_emb[our_id] = entity_emb[pykeen_id]
            # else: stays zero vector
        entity_emb = final_emb

    # Save embeddings
    emb_path = os.path.join(sp_kg_dir, "entity_embeddings.npy")
    np.save(emb_path, entity_emb)

    kg_summary = {
        "species": species,
        "drug": drug_name,
        "n_genes": len(genes_list),
        "n_entities": len(entity_to_id),
        "n_relations": len(relation_to_id),
        "n_triples": len(triples),
        "n_mechanism_classes": len(unique_mech),
        "entity_emb_shape": list(entity_emb.shape),
        "genes": genes_list[:20],
    }
    with open(os.path.join(sp_kg_dir, "kg_summary.json"), "w") as f:
        json.dump(kg_summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"  Entity embeddings: {entity_emb.shape}")
    print(f"  → {sp_kg_dir}/")
    print(f"  Time: {elapsed:.1f}s")

print("\nStage M5 complete.")
