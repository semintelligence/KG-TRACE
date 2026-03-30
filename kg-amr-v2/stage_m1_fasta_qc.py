"""
Stage M1: FASTA Quality Control
Validates downloaded BV-BRC FASTAs and saves valid genome-ID lists.
Output: data/download_logs/{species}_valid_fastas.txt
"""
import os, sys, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR

FASTA_BASE = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/fasta")
LOG_DIR    = os.path.join(PROJECT_DIR, "data/download_logs")
os.makedirs(LOG_DIR, exist_ok=True)

MIN_SIZE_BYTES = 100_000   # < 100 KB = likely empty / failed
MAX_SIZE_BYTES = 50_000_000  # > 50 MB = suspiciously large

SPECIES_LIST = [
    "Ecoli_ampicillin",
    "Kpneumoniae_cipro",
    "Kpneumoniae_carbapenem",
    "Abaumannii_carbapenem",
]

summary = {}

for species in SPECIES_LIST:
    fasta_dir = Path(FASTA_BASE) / species
    if not fasta_dir.exists():
        print(f"[WARN] {species}: directory not found — {fasta_dir}")
        summary[species] = {"valid": 0, "invalid": 0}
        continue

    valid_ids   = []
    reasons     = {}

    for fna_path in sorted(fasta_dir.glob("*.fna")):
        gid  = fna_path.stem
        size = fna_path.stat().st_size

        if size < MIN_SIZE_BYTES:
            reasons[gid] = f"too_small ({size} bytes)"
            continue
        if size > MAX_SIZE_BYTES:
            reasons[gid] = f"too_large ({size} bytes)"
            continue

        # Peek at first bytes — must start with ">"
        with open(fna_path, "rb") as f:
            header = f.read(200)
        if b">" not in header:
            reasons[gid] = "missing_fasta_header"
            continue

        valid_ids.append(gid)

    invalid_count = len(reasons)
    print(f"{species}: {len(valid_ids):4d} valid  |  {invalid_count:3d} invalid"
          + (f"  (top reason: {list(reasons.values())[0]})" if reasons else ""))

    out_path = os.path.join(LOG_DIR, f"{species}_valid_fastas.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(valid_ids))

    if reasons:
        bad_path = os.path.join(LOG_DIR, f"{species}_qc_rejected.json")
        with open(bad_path, "w") as f:
            json.dump(reasons, f, indent=2)

    summary[species] = {"valid": len(valid_ids), "invalid": invalid_count}

print("\n=== QC Summary ===")
total_valid = sum(v["valid"] for v in summary.values())
total_bad   = sum(v["invalid"] for v in summary.values())
for sp, s in summary.items():
    print(f"  {sp:<28} {s['valid']:4d} valid  {s['invalid']:3d} invalid")
print(f"  {'TOTAL':<28} {total_valid:4d} valid  {total_bad:3d} invalid")
print("\nStage M1 complete.")
