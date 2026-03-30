#!/usr/bin/env python3
"""
data/download_bvbrc.py — Robust BV-BRC genome FASTA downloader.
Reads genome IDs from Mendeley label files, downloads FASTA from BV-BRC.
Fully resume-safe. Logs every failure. Never fakes success counts.
"""
import os, time, glob, requests, pandas as pd
from pathlib import Path

PROJECT       = os.path.expanduser("~/Desktop/AMR NamanXSarika/kg-amr-v2")
GENOME_ID_DIR = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/genome_id")
FASTA_BASE    = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/fasta")
LOG_DIR       = os.path.join(PROJECT, "data/download_logs")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FASTA_BASE, exist_ok=True)

# BV-BRC REST API (FTP https endpoint does not resolve — API is the correct method)
BVBRC_API = (
    "https://www.bv-brc.org/api/genome_sequence/"
    "?eq(genome_id,{gid})&http_accept=application/dna+fasta"
)

BATCH_SIZE            = 50
SLEEP_BETWEEN_BATCHES = 2.0   # seconds
SLEEP_BETWEEN_REQS    = 0.3   # seconds  (polite rate limit)
MIN_FASTA_BYTES       = 500   # below this → treat as failed/empty


def load_genome_ids(txt_path):
    """Parse Mendeley label file. Returns list of (genome_id_str, label) tuples."""
    # Format: tab-sep, unnamed index col, then genome_id, then resistant_phenotype
    df = pd.read_csv(txt_path, sep="\t", index_col=0)
    if "genome_id" not in df.columns:
        raise ValueError(f"Expected 'genome_id' column in {txt_path}, got {list(df.columns)}")
    ids = df["genome_id"].astype(str).tolist()
    labels = df["resistant_phenotype"].astype(int).tolist()
    return list(zip(ids, labels))


def is_valid_fasta(path):
    """Check a file looks like a real FASTA (non-empty, starts with '>')."""
    try:
        if os.path.getsize(path) < MIN_FASTA_BYTES:
            return False
        with open(path, "rb") as f:
            first = f.read(10)
        return b">" in first
    except Exception:
        return False


def download_genome(genome_id, out_path, session):
    """
    Try FTP URL first, fall back to REST API.
    Returns (success: bool, bytes_written: int, method: str, error: str).
    """
    if is_valid_fasta(out_path):
        return True, os.path.getsize(out_path), "cached", ""

    last_error = ""
    for method, url in [
        ("api", BVBRC_API.format(gid=genome_id)),
    ]:
        try:
            resp = session.get(url, timeout=90, stream=False)
            if resp.status_code == 200:
                content = resp.content
                # Validate: must look like FASTA
                if len(content) >= MIN_FASTA_BYTES and b">" in content[:200]:
                    with open(out_path, "wb") as f:
                        f.write(content)
                    return True, len(content), method, ""
                else:
                    last_error = (
                        f"{method} HTTP 200 but invalid/empty FASTA "
                        f"(got {len(content)} bytes, preview={content[:50]!r})"
                    )
            else:
                last_error = f"{method} HTTP {resp.status_code} from {url}"
        except requests.exceptions.Timeout:
            last_error = f"{method} timeout after 90s"
        except requests.exceptions.ConnectionError as e:
            last_error = f"{method} connection error: {e}"
        except Exception as e:
            last_error = f"{method} unexpected error: {e}"
        time.sleep(0.2)  # brief pause between FTP→API fallback

    return False, 0, "failed", last_error


def download_species(txt_file, species_name):
    """Download all genomes for one species-antibiotic pair. Returns (total_on_disk, fail_count)."""
    print(f"\n{'='*60}")
    print(f"Species: {species_name}")
    print(f"Source:  {txt_file}")

    pairs = load_genome_ids(txt_file)
    genome_ids = [gid for gid, _ in pairs]
    print(f"Genome IDs in label file: {len(genome_ids)}")

    out_dir = os.path.join(FASTA_BASE, species_name)
    os.makedirs(out_dir, exist_ok=True)

    log_failed = os.path.join(LOG_DIR, f"{species_name}_failed.txt")
    log_summary = os.path.join(LOG_DIR, f"{species_name}_summary.txt")

    # Resume-safe: find already-valid FASTAs
    existing = {
        f.replace(".fna", "")
        for f in os.listdir(out_dir)
        if f.endswith(".fna") and is_valid_fasta(os.path.join(out_dir, f))
    }
    remaining = [g for g in genome_ids if g not in existing]
    print(f"Already cached (valid):   {len(existing)}")
    print(f"Remaining to download:    {len(remaining)}")

    if not remaining:
        print("✅ All genomes already downloaded — nothing to do.")
        return len(existing), 0

    success_count = 0
    fail_count = 0
    failed_records = []  # (genome_id, error_msg)

    session = requests.Session()
    session.headers.update({"User-Agent": "KG-AMR-v2/1.0 (research; contact: github.com/kg-amr)"})

    for i, gid in enumerate(remaining):
        out_path = os.path.join(out_dir, f"{gid}.fna")
        ok, nbytes, method, err = download_genome(gid, out_path, session)

        if ok:
            success_count += 1
        else:
            fail_count += 1
            failed_records.append((gid, err))
            # Remove partial file if it exists but is invalid
            if os.path.exists(out_path) and not is_valid_fasta(out_path):
                os.remove(out_path)

        # Progress report every 50
        if (i + 1) % 50 == 0 or (i + 1) == len(remaining):
            total_so_far = len(existing) + success_count
            pct = total_so_far / len(genome_ids) * 100
            print(
                f"  [{i+1}/{len(remaining)}] "
                f"ok={success_count} fail={fail_count} "
                f"total_on_disk={total_so_far}/{len(genome_ids)} ({pct:.1f}%)"
            )

        # Save failed log every 100 genomes (progress checkpoint)
        if (i + 1) % 100 == 0:
            with open(log_failed, "w") as f:
                for fgid, ferr in failed_records:
                    f.write(f"{fgid}\t{ferr}\n")

        # Rate limiting
        if (i + 1) % BATCH_SIZE == 0:
            time.sleep(SLEEP_BETWEEN_BATCHES)
        else:
            time.sleep(SLEEP_BETWEEN_REQS)

    # Final writes
    with open(log_failed, "w") as f:
        for fgid, ferr in failed_records:
            f.write(f"{fgid}\t{ferr}\n")

    total_on_disk = len(existing) + success_count
    summary_lines = [
        f"Species:                   {species_name}",
        f"Total in label file:       {len(genome_ids)}",
        f"Previously cached:         {len(existing)}",
        f"Downloaded this session:   {success_count}",
        f"Failed this session:       {fail_count}",
        f"Total valid FASTAs on disk: {total_on_disk} / {len(genome_ids)}",
        f"Failed IDs logged to:       {log_failed}",
    ]
    with open(log_summary, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\n=== {species_name} DOWNLOAD SUMMARY ===")
    for line in summary_lines:
        print(f"  {line}")
    print(f"\nDownloaded {total_on_disk} / {len(genome_ids)} genomes successfully")
    return total_on_disk, fail_count


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import sys

    # ── Step 1: List all available genome ID files ───────────────────────────
    all_txt = sorted(glob.glob(os.path.join(GENOME_ID_DIR, "Data_*.txt")))
    print("All available genome ID files:")
    for f in all_txt:
        try:
            n = sum(1 for _ in open(f)) - 1  # subtract header
            print(f"  {os.path.basename(f)} — {n} genomes")
        except Exception as e:
            print(f"  {os.path.basename(f)} — ERROR: {e}")

    # ── Step 2: Build download targets (priority order) ──────────────────────
    all_basenames = {os.path.basename(f): f for f in all_txt}

    targets = {}  # species_tag → (full_txt_path, description)

    # Priority 1: E. coli + Ampicillin
    f = "Data_Escherichia_coli_ampicillin.txt"
    if f in all_basenames:
        targets["Ecoli_ampicillin"] = all_basenames[f]

    # Priority 2: K. pneumoniae + Cipro
    f = "Data_Klebsiella_pneumoniae_ciprofloxacin.txt"
    if f in all_basenames:
        targets["Kpneumoniae_cipro"] = all_basenames[f]

    # Priority 3: K. pneumoniae + carbapenem (meropenem or imipenem)
    for candidate in ["Data_Klebsiella_pneumoniae_meropenem.txt",
                      "Data_Klebsiella_pneumoniae_imipenem.txt",
                      "Data_Klebsiella_pneumoniae_carbapenem.txt"]:
        if candidate in all_basenames:
            targets["Kpneumoniae_carbapenem"] = all_basenames[candidate]
            break

    # Priority 4: A. baumannii + carbapenem (meropenem or imipenem)
    for candidate in ["Data_Acinetobacter_baumannii_meropenem.txt",
                      "Data_Acinetobacter_baumannii_imipenem.txt",
                      "Data_Acinetobacter_baumannii_carbapenem.txt"]:
        if candidate in all_basenames:
            targets["Abaumannii_carbapenem"] = all_basenames[candidate]
            break

    print(f"\nDownload targets identified ({len(targets)}):")
    for tag, path in targets.items():
        print(f"  {tag}  ←  {os.path.basename(path)}")

    if not targets:
        print("❌ No targets found — check GENOME_ID_DIR path")
        sys.exit(1)

    # ── Step 3: Download each target in priority order ───────────────────────
    overall_results = {}
    for species_name, txt_path in targets.items():
        total, fails = download_species(txt_path, species_name)
        overall_results[species_name] = {"total": total, "failed": fails}

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== FINAL DOWNLOAD SUMMARY (all species) ===")
    grand_total_ok = 0
    grand_total_fail = 0
    for species, r in overall_results.items():
        print(f"  {species:30s}: {r['total']:5d} downloaded, {r['failed']:5d} failed")
        grand_total_ok   += r["total"]
        grand_total_fail += r["failed"]
    print(f"\n  GRAND TOTAL: {grand_total_ok} downloaded, {grand_total_fail} failed")
    print(f"  Logs: {LOG_DIR}")
