"""
Task 4 Extension 1 — Mendeley Multi-Species Data Report.

HONEST ASSESSMENT: The Mendeley dataset files contain only genome IDs and
binary phenotype labels (R/S). Genome sequences (FASTA/FASTQ) are NOT
available locally, so k-mer TF-IDF features CANNOT be computed.
Additionally, the KG and RotatE embeddings are MTB-specific (26 WHO
catalogue genes) and not applicable to non-TB species.

This script reports dataset statistics for the requested Mendeley pairs
and documents the limitation.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import MENDELEY_DIR, PROJECT_DIR

EVALUATE_DIR = os.path.join(PROJECT_DIR, "evaluate")
os.makedirs(EVALUATE_DIR, exist_ok=True)

GENOME_ID_DIR = os.path.join(MENDELEY_DIR, "genome_id")

# Requested pairs — map to actual filenames.
# "carbapenem" mapped to meropenem (a carbapenem antibiotic) since no
# generic "carbapenem" file exists.
MENDELEY_PAIRS = [
    {
        "species": "Klebsiella pneumoniae",
        "antibiotic": "ciprofloxacin",
        "file": "Data_Klebsiella_pneumoniae_ciprofloxacin.txt",
    },
    {
        "species": "Klebsiella pneumoniae",
        "antibiotic": "meropenem",
        "file": "Data_Klebsiella_pneumoniae_meropenem.txt",
        "note": "Meropenem used as carbapenem proxy",
    },
    {
        "species": "Acinetobacter baumannii",
        "antibiotic": "meropenem",
        "file": "Data_Acinetobacter_baumannii_meropenem.txt",
        "note": "Meropenem used as carbapenem proxy",
    },
    {
        "species": "Escherichia coli",
        "antibiotic": "ampicillin",
        "file": "Data_Escherichia_coli_ampicillin.txt",
    },
]

print("=" * 60)
print("  Mendeley Multi-Species Data Report")
print("=" * 60)

report = {
    "limitation": (
        "Genome sequences are not available locally for Mendeley species. "
        "Only genome IDs and phenotype labels are present. "
        "K-mer TF-IDF features cannot be computed without genome assemblies. "
        "The existing KG and RotatE embeddings are M. tuberculosis-specific "
        "(26 WHO catalogue genes) and do not apply to non-TB organisms."
    ),
    "pairs": [],
}

for pair in MENDELEY_PAIRS:
    fname = pair["file"]
    fpath = os.path.join(GENOME_ID_DIR, fname)

    entry = {
        "species": pair["species"],
        "antibiotic": pair["antibiotic"],
        "file": fname,
    }
    if pair.get("note"):
        entry["note"] = pair["note"]

    if not os.path.exists(fpath):
        entry["status"] = "file_not_found"
        entry["n_samples"] = 0
        print(f"\n  {pair['species']} + {pair['antibiotic']}: FILE NOT FOUND")
    else:
        n_total = 0
        n_r = 0
        n_s = 0
        with open(fpath) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    pheno = int(parts[-1])
                    if pheno == 1:
                        n_r += 1
                    else:
                        n_s += 1
                    n_total += 1

        entry["status"] = "labels_only"
        entry["n_samples"] = n_total
        entry["n_resistant"] = n_r
        entry["n_susceptible"] = n_s
        entry["R_pct"] = round(100 * n_r / n_total, 1) if n_total > 0 else 0.0

        print(f"\n  {pair['species']} + {pair['antibiotic']}:")
        print(f"    Samples: {n_total} ({n_r} R / {n_s} S, {entry['R_pct']}% R)")
        print(f"    Status: labels only — no features available")

    report["pairs"].append(entry)

# Save report
report_path = os.path.join(EVALUATE_DIR, "mendeley_data_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n  Saved to: {report_path}")
print("\n  CONCLUSION: Mendeley extension requires genome sequences for")
print("  k-mer feature computation. These are not available locally.")
print("  The KG-Trace multi-dataset evaluation uses MTB CRyPTIC data")
print("  across multiple drugs instead.")
print("DONE — run_mendeley_extension.py")
