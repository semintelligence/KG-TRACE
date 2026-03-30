#!/usr/bin/env python3
"""
data/download_bvbrc_fast.py — Concurrent BV-BRC genome FASTA downloader.
Uses ThreadPoolExecutor for parallel HTTP requests (~20x faster than sequential).
Resume-safe: skips already-downloaded FASTAs. Logs every failure honestly.
"""
import os, glob, time, threading, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT       = os.path.expanduser("~/Desktop/AMR NamanXSarika/kg-amr-v2")
GENOME_ID_DIR = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/genome_id")
FASTA_BASE    = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/fasta")
LOG_DIR       = os.path.join(PROJECT, "data/download_logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FASTA_BASE, exist_ok=True)

BVBRC_API = (
    "https://www.bv-brc.org/api/genome_sequence/"
    "?eq(genome_id,{gid})&http_accept=application/dna+fasta"
)

MAX_WORKERS   = 20     # concurrent threads — polite but fast
MIN_FASTA_BYTES = 500


def load_genome_ids(txt_path):
    df = pd.read_csv(txt_path, sep="\t", index_col=0)
    return df["genome_id"].astype(str).tolist()


def is_valid_fasta(path):
    try:
        if os.path.getsize(path) < MIN_FASTA_BYTES:
            return False
        with open(path, "rb") as f:
            return b">" in f.read(20)
    except Exception:
        return False


def download_one(genome_id, out_path, session):
    """Download a single genome. Returns (genome_id, success, bytes, error)."""
    if is_valid_fasta(out_path):
        return genome_id, True, os.path.getsize(out_path), "cached"
    url = BVBRC_API.format(gid=genome_id)
    try:
        resp = session.get(url, timeout=90)
        if resp.status_code == 200:
            content = resp.content
            if len(content) >= MIN_FASTA_BYTES and b">" in content[:200]:
                with open(out_path, "wb") as f:
                    f.write(content)
                return genome_id, True, len(content), ""
            else:
                return genome_id, False, 0, f"HTTP 200 but invalid FASTA ({len(content)} bytes)"
        else:
            return genome_id, False, 0, f"HTTP {resp.status_code}"
    except requests.exceptions.Timeout:
        return genome_id, False, 0, "timeout"
    except Exception as e:
        return genome_id, False, 0, str(e)


def download_species(txt_file, species_name):
    """Download all genomes for one species concurrently."""
    print(f"\n{'='*60}")
    print(f"Species: {species_name}")

    genome_ids = load_genome_ids(txt_file)
    print(f"Total in label file: {len(genome_ids)}")

    out_dir = os.path.join(FASTA_BASE, species_name)
    os.makedirs(out_dir, exist_ok=True)

    # Resume: skip already-valid FASTAs
    existing = {
        f.replace(".fna", "")
        for f in os.listdir(out_dir)
        if f.endswith(".fna") and is_valid_fasta(os.path.join(out_dir, f))
    }
    remaining = [g for g in genome_ids if g not in existing]
    print(f"Already cached: {len(existing)}  |  Remaining: {len(remaining)}")

    if not remaining:
        print("All genomes already downloaded.")
        return len(existing), 0

    log_failed  = os.path.join(LOG_DIR, f"{species_name}_failed.txt")
    log_summary = os.path.join(LOG_DIR, f"{species_name}_summary.txt")

    # Thread-safe counters
    lock = threading.Lock()
    success_count = [0]
    fail_count    = [0]
    failed_records = []
    start_time = time.time()

    # Build a shared session pool (one per thread via thread-local)
    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": "KG-AMR-v2/1.0 (research)"})
            thread_local.session = s
        return thread_local.session

    def worker(gid):
        out_path = os.path.join(out_dir, f"{gid}.fna")
        gid_ret, ok, nbytes, err = download_one(gid, out_path, get_session())
        with lock:
            if ok:
                success_count[0] += 1
            else:
                fail_count[0] += 1
                failed_records.append((gid, err))
                # Remove corrupt partial file
                if os.path.exists(out_path) and not is_valid_fasta(out_path):
                    try:
                        os.remove(out_path)
                    except Exception:
                        pass
            done = success_count[0] + fail_count[0]
            # Progress every 100 completions
            if done % 100 == 0:
                total_disk = len(existing) + success_count[0]
                elapsed = time.time() - start_time
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta_min = (len(remaining) - done) / (done / elapsed) / 60 if done > 0 else 0
                print(
                    f"  [{done}/{len(remaining)}] "
                    f"ok={success_count[0]} fail={fail_count[0]} "
                    f"on_disk={total_disk}/{len(genome_ids)} "
                    f"rate={rate:.0f}/min  ETA={eta_min:.0f}min"
                )
                # Checkpoint: flush failed log
                with open(log_failed, "w") as f:
                    for fg, fe in failed_records:
                        f.write(f"{fg}\t{fe}\n")

    print(f"Starting {MAX_WORKERS} concurrent workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, gid): gid for gid in remaining}
        # Wait for all; progress is printed inside worker
        for fut in as_completed(futures):
            pass  # errors already handled inside worker

    # Final logs
    with open(log_failed, "w") as f:
        for fg, fe in failed_records:
            f.write(f"{fg}\t{fe}\n")

    total_on_disk = len(existing) + success_count[0]
    elapsed = time.time() - start_time
    summary = [
        f"Species:                    {species_name}",
        f"Total in label file:        {len(genome_ids)}",
        f"Previously cached:          {len(existing)}",
        f"Downloaded this session:    {success_count[0]}",
        f"Failed this session:        {fail_count[0]}",
        f"Total valid FASTAs on disk: {total_on_disk} / {len(genome_ids)}",
        f"Elapsed:                    {elapsed/60:.1f} min",
        f"Average rate:               {len(remaining)/elapsed*60:.0f} genomes/min",
        f"Failed IDs logged to:       {log_failed}",
    ]
    with open(log_summary, "w") as f:
        f.write("\n".join(summary) + "\n")

    print(f"\n=== {species_name} SUMMARY ===")
    for line in summary:
        print(f"  {line}")
    print(f"\nDownloaded {total_on_disk} / {len(genome_ids)} genomes successfully")
    return total_on_disk, fail_count[0]


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_txt = {os.path.basename(f): f for f in sorted(
        glob.glob(os.path.join(GENOME_ID_DIR, "Data_*.txt"))
    )}

    # Priority targets
    targets = {}
    for key, candidates in [
        ("Ecoli_ampicillin",       ["Data_Escherichia_coli_ampicillin.txt"]),
        ("Kpneumoniae_cipro",      ["Data_Klebsiella_pneumoniae_ciprofloxacin.txt"]),
        ("Kpneumoniae_carbapenem", ["Data_Klebsiella_pneumoniae_meropenem.txt",
                                    "Data_Klebsiella_pneumoniae_imipenem.txt"]),
        ("Abaumannii_carbapenem",  ["Data_Acinetobacter_baumannii_meropenem.txt",
                                    "Data_Acinetobacter_baumannii_imipenem.txt"]),
    ]:
        for c in candidates:
            if c in all_txt:
                targets[key] = all_txt[c]
                break

    print(f"Download targets ({len(targets)}):")
    for tag, path in targets.items():
        n = sum(1 for _ in open(path)) - 1
        print(f"  {tag:30s} {n:5d} genomes  [{os.path.basename(path)}]")

    overall = {}
    for species_name, txt_path in targets.items():
        total, fails = download_species(txt_path, species_name)
        overall[species_name] = {"total": total, "failed": fails}

    print(f"\n{'='*60}")
    print("=== FINAL SUMMARY ===")
    grand_ok = grand_fail = 0
    for sp, r in overall.items():
        print(f"  {sp:30s}: {r['total']:5d} ok, {r['failed']:5d} failed")
        grand_ok   += r["total"]
        grand_fail += r["failed"]
    print(f"\n  GRAND TOTAL: {grand_ok} downloaded, {grand_fail} failed")
    print(f"  Logs: {LOG_DIR}")
