"""
Stage M3: BV-BRC AMR Gene Annotations
Fetches resistance gene presence data for all valid genomes via BV-BRC REST API.
Builds a binary gene-presence matrix per species.

Outputs per species:
  features/annotations/{species}_gene_presence.npz  (binary float32 sparse matrix)
  features/annotations/{species}_gene_ids.npy        (genome IDs array)
  features/annotations/{species}_genes.json          (gene name → column index)
  features/annotations/{species}_amr_raw.json        (raw API records)
"""
import os, sys, json, time
from pathlib import Path
from collections import defaultdict

import requests
import numpy as np
import scipy.sparse as sp
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR

# ── Config ─────────────────────────────────────────────────────────────────
LOG_DIR    = os.path.join(PROJECT_DIR, "data/download_logs")
ANNOT_DIR  = os.path.join(PROJECT_DIR, "features/annotations")
os.makedirs(ANNOT_DIR, exist_ok=True)

BVBRC_SP_GENE_URL = "https://www.bv-brc.org/api/sp_gene/"
BATCH_SIZE    = 50     # genome IDs per API call (sp_gene returns more data)
MAX_WORKERS   = 5      # concurrent API requests
REQUEST_TIMEOUT = 45   # seconds

SPECIES_LIST = [
    "Ecoli_ampicillin",
    "Kpneumoniae_cipro",
    "Kpneumoniae_carbapenem",
    "Abaumannii_carbapenem",
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def fetch_amr_batch(genome_ids):
    """Fetch CARD antibiotic-resistance gene records for a batch of genome IDs."""
    gid_str = ",".join(genome_ids)
    # Use sp_gene endpoint filtered to CARD Antibiotic Resistance entries.
    # The multi-word property value must be quoted in BV-BRC RQL.
    url = (f'{BVBRC_SP_GENE_URL}'
           f'?and(in(genome_id,({gid_str})),eq(property,"Antibiotic Resistance"))'
           f'&select(genome_id,gene,product,source)'
           f'&limit(50000)&http_accept=application/json')
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        return None, str(e)
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        records = resp.json()
        return records, None
    except Exception as e:
        return None, str(e)


# ── Per-species processing ──────────────────────────────────────────────────
for species in SPECIES_LIST:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Processing: {species}")

    valid_file = os.path.join(LOG_DIR, f"{species}_valid_fastas.txt")
    if not os.path.exists(valid_file):
        print(f"  [SKIP] No valid FASTA list")
        continue
    with open(valid_file) as f:
        genome_ids = [l.strip() for l in f if l.strip()]
    N = len(genome_ids)
    print(f"  Genomes to annotate: {N}")

    # Check if raw records already cached
    raw_cache = os.path.join(ANNOT_DIR, f"{species}_amr_raw.json")
    if os.path.exists(raw_cache):
        print(f"  Loading cached AMR records...")
        with open(raw_cache) as f:
            all_records = json.load(f)
        print(f"  Loaded {len(all_records):,} cached records")
    else:
        # Fetch in batches with threading
        batches = [genome_ids[i:i+BATCH_SIZE] for i in range(0, N, BATCH_SIZE)]
        print(f"  Fetching {len(batches)} batches ({BATCH_SIZE} genomes each) "
              f"with {MAX_WORKERS} workers...")
        all_records = []
        n_errors = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            future_to_batch = {exe.submit(fetch_amr_batch, b): i
                               for i, b in enumerate(batches)}
            for future in as_completed(future_to_batch):
                bi = future_to_batch[future]
                records, err = future.result()
                if err:
                    n_errors += 1
                    print(f"    [WARN] Batch {bi}: {err}")
                else:
                    all_records.extend(records)
                if (bi + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"    [{bi+1}/{len(batches)} batches]  "
                          f"{len(all_records):,} records  ({elapsed:.0f}s)")

        print(f"  Total records fetched: {len(all_records):,}  ({n_errors} batch errors)")
        with open(raw_cache, "w") as f:
            json.dump(all_records, f)
        print(f"  Cached to: {raw_cache}")

    # ── Build gene presence matrix ─────────────────────────────────────────
    # Extract unique, non-empty gene names
    gene_set = set()
    genome_gene_map = defaultdict(set)  # gid → set of gene names
    for rec in all_records:
        gid  = str(rec.get("genome_id", "")).strip()
        gene = str(rec.get("gene", "")).strip()
        if gid and gene and gene not in ("", "None", "null"):
            gene_set.add(gene)
            genome_gene_map[gid].add(gene)

    genes_list = sorted(gene_set)
    gene_to_col = {g: i for i, g in enumerate(genes_list)}
    G = len(genes_list)
    print(f"  Unique resistance genes: {G}")
    print(f"  Genomes with ≥1 gene:    {len(genome_gene_map)}")
    print(f"  Genomes with 0 genes:    {N - len(genome_gene_map)}")

    if G == 0:
        print(f"  [WARN] No gene annotations found — creating zero matrix")
        G = 1
        genes_list = ["no_gene"]
        gene_to_col = {"no_gene": 0}

    # Build binary gene presence matrix
    rows, cols = [], []
    gid_to_row = {gid: i for i, gid in enumerate(genome_ids)}
    for gid, gene_names in genome_gene_map.items():
        row = gid_to_row.get(gid)
        if row is None:
            continue
        for gene in gene_names:
            col = gene_to_col.get(gene)
            if col is not None:
                rows.append(row)
                cols.append(col)

    data = np.ones(len(rows), dtype=np.float32)
    gene_pres = sp.csr_matrix((data, (rows, cols)), shape=(N, G), dtype=np.float32)

    # ── Save ───────────────────────────────────────────────────────────────
    sp.save_npz(os.path.join(ANNOT_DIR, f"{species}_gene_presence.npz"), gene_pres)
    np.save(os.path.join(ANNOT_DIR, f"{species}_gene_ids.npy"), np.array(genome_ids))
    with open(os.path.join(ANNOT_DIR, f"{species}_genes.json"), "w") as f:
        json.dump(gene_to_col, f, indent=2)

    # Print top-20 most common genes
    gene_counts = np.asarray(gene_pres.sum(axis=0)).flatten()
    top_genes = sorted(zip(gene_counts, genes_list), reverse=True)[:20]
    print(f"  Top genes by prevalence:")
    for cnt, g in top_genes:
        print(f"    {g:<30} {int(cnt):4d}/{N} genomes ({100*cnt/N:.1f}%)")

    elapsed = time.time() - t0
    print(f"  Matrix: {gene_pres.shape}  nnz={gene_pres.nnz}")
    print(f"  Time: {elapsed:.1f}s")

print("\nStage M3 complete.")
