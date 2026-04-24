"""
Stage M2: K-mer TF-IDF Feature Extraction
Two-pass memory-efficient extraction of k=21 features from FASTA files.
No jellyfish / external tools required — pure Python + scipy sparse matrices.

Pass 1: Count document frequency (how many genomes have each 21-mer).
Pass 2: Compute per-genome TF, apply IDF, build CSR sparse matrix.

Outputs per species:
  features/matrices/{species}_kmer.npz        (sparse float32 TF-IDF matrix)
  features/matrices/{species}_kmer_ids.npy     (genome IDs array)
  features/matrices/{species}_kmer_vocab.json  (kmer → column index)
"""
import os, sys, json, math, time
from pathlib import Path
from collections import Counter

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR, MENDELEY_DIR

# ── Config ───────────────────────────────────────────────────────────────────
FASTA_BASE  = os.path.join(MENDELEY_DIR, "fasta")
LOG_DIR     = os.path.join(PROJECT_DIR, "data/download_logs")
MATRIX_DIR  = os.path.join(PROJECT_DIR, "features/matrices")
os.makedirs(MATRIX_DIR, exist_ok=True)

K          = 21       # k-mer length
STRIDE     = 21       # non-overlapping (memory-efficient)
MIN_DF     = 5        # minimum number of genomes a k-mer must appear in
MAX_DF_FRAC= 0.95     # maximum fraction of genomes (to filter core-genome k-mers)
MAX_FEAT   = 100_000  # cap on vocabulary size (top by IDF)

SPECIES_LIST = [
    "Ecoli_ampicillin",
    "Kpneumoniae_cipro",
    "Kpneumoniae_carbapenem",
    "Abaumannii_carbapenem",
]

# ── Helpers ─────────────────────────────────────────────────────────────────
def read_fasta_seq(path):
    """Concatenate all contig sequences from a FASTA file (strips headers)."""
    seqs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seqs.append(line.upper())
    return "".join(seqs)


def extract_kmers_set(seq, k, stride):
    """Return set of distinct k-mers (for document frequency counting)."""
    return {seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)
            if "N" not in seq[i:i+k]}


def extract_kmers_counter(seq, k, stride):
    """Return Counter of k-mer term frequencies."""
    return Counter(seq[i:i+k] for i in range(0, len(seq) - k + 1, stride)
                   if "N" not in seq[i:i+k])


# ── Per-species extraction ──────────────────────────────────────────────────
for species in SPECIES_LIST:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Processing: {species}")

    # Load valid genome IDs
    valid_file = os.path.join(LOG_DIR, f"{species}_valid_fastas.txt")
    if not os.path.exists(valid_file):
        print(f"  [SKIP] No valid FASTA list — run stage_m1 first")
        continue
    with open(valid_file) as f:
        genome_ids = [l.strip() for l in f if l.strip()]
    N = len(genome_ids)
    print(f"  Genomes: {N}")

    fasta_dir = Path(FASTA_BASE) / species

    # ── Pass 1: Document frequency ─────────────────────────────────────────
    print(f"  Pass 1: counting document frequencies (k={K}, stride={STRIDE})...")
    doc_freq = Counter()
    for gi, gid in enumerate(genome_ids):
        fna = fasta_dir / f"{gid}.fna"
        if not fna.exists():
            continue
        seq = read_fasta_seq(fna)
        kmers = extract_kmers_set(seq, K, STRIDE)
        doc_freq.update(kmers)
        if (gi + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{gi+1}/{N}]  {len(doc_freq):,} unique k-mers  ({elapsed:.0f}s)")

    print(f"  Total unique k-mers observed: {len(doc_freq):,}")

    # ── Build vocabulary ───────────────────────────────────────────────────
    max_df_count = int(MAX_DF_FRAC * N)
    print(f"  Filtering: min_df={MIN_DF}, max_df={max_df_count} ({MAX_DF_FRAC*100:.0f}%)")

    eligible = {kmer: cnt for kmer, cnt in doc_freq.items()
                if MIN_DF <= cnt <= max_df_count}
    print(f"  Eligible after df filter: {len(eligible):,}")

    # Sort by IDF descending (rarest but present are most informative)
    # IDF = log((N + 1) / (df + 1)) + 1  (sklearn smooth IDF)
    idf_vals = {kmer: math.log((N + 1) / (cnt + 1)) + 1
                for kmer, cnt in eligible.items()}
    top_kmers = sorted(idf_vals, key=lambda k: idf_vals[k], reverse=True)[:MAX_FEAT]
    vocab = {kmer: idx for idx, kmer in enumerate(top_kmers)}
    V = len(vocab)
    print(f"  Vocabulary size: {V:,}")

    # ── Pass 2: Build TF-IDF sparse matrix ────────────────────────────────
    print(f"  Pass 2: building TF-IDF matrix ({N} x {V})...")
    rows, cols, data_vals = [], [], []
    idf_array = np.array([idf_vals[k] for k in top_kmers], dtype=np.float32)

    valid_indices = []
    for gi, gid in enumerate(genome_ids):
        fna = fasta_dir / f"{gid}.fna"
        if not fna.exists():
            continue

        seq = read_fasta_seq(fna)
        tf_counter = extract_kmers_counter(seq, K, STRIDE)
        total_tokens = sum(tf_counter.values())
        if total_tokens == 0:
            continue

        for kmer, raw_tf in tf_counter.items():
            col = vocab.get(kmer)
            if col is None:
                continue
            # sublinear TF: 1 + log(tf)
            tf_val = 1.0 + math.log(raw_tf)
            rows.append(len(valid_indices))
            cols.append(col)
            data_vals.append(tf_val)

        valid_indices.append(gi)
        if (len(valid_indices)) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{len(valid_indices)}/{N}] ({elapsed:.0f}s)")

    # Stack into sparse matrix and apply IDF
    n_valid = len(valid_indices)
    X = sp.csr_matrix((data_vals, (rows, cols)), shape=(n_valid, V), dtype=np.float32)
    # Multiply each column by its IDF
    X = X.multiply(idf_array[np.newaxis, :])
    # L2-normalise rows (standard cosine space)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    X = X.multiply(1.0 / norms[:, np.newaxis])
    X = X.tocsr().astype(np.float32)

    actual_gids = np.array([genome_ids[i] for i in valid_indices])

    # ── Save ───────────────────────────────────────────────────────────────
    out_npz  = os.path.join(MATRIX_DIR, f"{species}_kmer.npz")
    out_ids  = os.path.join(MATRIX_DIR, f"{species}_kmer_ids.npy")
    out_vocab= os.path.join(MATRIX_DIR, f"{species}_kmer_vocab.json")

    sp.save_npz(out_npz, X)
    np.save(out_ids, actual_gids)
    with open(out_vocab, "w") as f:
        json.dump({km: int(idx) for km, idx in vocab.items()}, f)

    elapsed = time.time() - t0
    print(f"  Saved: {X.shape[0]} genomes × {X.shape[1]} features  "
          f"  nnz={X.nnz:,}  density={X.nnz/(X.shape[0]*X.shape[1]):.3%}")
    print(f"  → {out_npz}")
    print(f"  Time: {elapsed:.1f}s")

print("\nStage M2 complete.")
