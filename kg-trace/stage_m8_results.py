"""
Stage M8: Update Results CSV + Dashboard HTML
Merges per-species test results with existing MTB results.
Appends new rows to evaluate/final_results.csv and updates explain/report.html.

Reads:
  model/species/{species}/test_results.json  (from Stage M6)
  evaluate/final_results.csv                 (existing MTB results)

Writes:
  evaluate/final_results.csv                 (updated, deduplicated)
  explain/report.html                        (updated dashboard)
"""
import os, sys, json
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR

EVAL_DIR    = os.path.join(PROJECT_DIR, "evaluate")
MODEL_BASE  = os.path.join(PROJECT_DIR, "model/species")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")

SPECIES_NICE = {
    "Ecoli_ampicillin":       "E. coli / Ampicillin",
    "Kpneumoniae_cipro":      "K. pneumoniae / Ciprofloxacin",
    "Kpneumoniae_carbapenem": "K. pneumoniae / Carbapenem",
    "Abaumannii_carbapenem":  "A. baumannii / Carbapenem",
}

SPECIES_LIST = list(SPECIES_NICE.keys())

# ── Load new results ─────────────────────────────────────────────────────────
new_rows = []
for species in SPECIES_LIST:
    rpath = os.path.join(MODEL_BASE, species, "test_results.json")
    if not os.path.exists(rpath):
        print(f"[SKIP] No results for {species}")
        continue
    with open(rpath) as f:
        r = json.load(f)
    new_rows.append({
        "dataset":        SPECIES_NICE[species],
        "species_key":    species,
        "source":         "Mendeley/BV-BRC",
        "antibiotic":     species.split("_", 1)[1],
        "n_train":        r.get("n_train", ""),
        "n_val":          r.get("n_val", ""),
        "n_test":         r.get("n_test", ""),
        "f1_macro":       r.get("f1_macro", ""),
        "f1_R":           r.get("f1_R", ""),
        "f1_S":           r.get("f1_S", ""),
        "auc_roc":        r.get("auc_roc", ""),
        "accuracy":       r.get("accuracy", ""),
        "kmer_dim":       r.get("kmer_dim", ""),
        "model":          "KG-Trace",
    })
    print(f"  Loaded: {species}  F1={r.get('f1_macro', 'N/A'):.4f}"
          if isinstance(r.get('f1_macro'), float) else f"  Loaded: {species}")

new_df = pd.DataFrame(new_rows)
print(f"\nNew species rows: {len(new_df)}")

# ── Save Mendeley results separately (different schema from MTB results) ───
mendeley_csv = os.path.join(EVAL_DIR, "mendeley_results.csv")
new_df.to_csv(mendeley_csv, index=False)
print(f"Saved: {mendeley_csv}")

# ── Use new_df for dashboard ────────────────────────────────────────────────
merged_df = new_df

# ── Update dashboard HTML ────────────────────────────────────────────────────
report_path = os.path.join(EXPLAIN_DIR, "report.html")

# Build a nice HTML table for the new species results
def _fmt(v, pct=False):
    if v == "" or v is None:
        return "—"
    try:
        f = float(v)
        if pct:
            return f"{f*100:.1f}%"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)

table_rows = ""
for _, row in merged_df.sort_values("dataset").iterrows():
    color = ""
    if "Mendeley" in str(row.get("source", "")):
        color = ' style="background:#f0fff4"'
    f1str = _fmt(row.get("f1_macro"))
    aucstr = _fmt(row.get("auc_roc"))
    accstr = _fmt(row.get("accuracy"))
    srcstr = str(row.get("source", ""))
    ntest  = str(row.get("n_test", "—"))
    table_rows += (
        f'<tr{color}>'
        f'<td>{row["dataset"]}</td>'
        f'<td>{srcstr}</td>'
        f'<td>{ntest}</td>'
        f'<td><b>{f1str}</b></td>'
        f'<td>{aucstr}</td>'
        f'<td>{accstr}</td>'
        f'</tr>\n'
    )

species_section_html = f"""
<section id="mendeley-results" style="margin-top:40px">
  <h2>Multi-Species Results (Mendeley Extension)</h2>
  <p style="color:#555">Species trained on BV-BRC genomic sequences with 21-mer TF-IDF features
  and species-specific RotatE KG embeddings from BV-BRC AMR annotations.</p>
  <table border="1" cellpadding="6" cellspacing="0"
         style="border-collapse:collapse;width:100%;font-family:monospace;font-size:13px">
    <thead style="background:#2c5282;color:white">
      <tr>
        <th>Dataset</th><th>Source</th><th>Test N</th>
        <th>F1-macro</th><th>AUC-ROC</th><th>Accuracy</th>
      </tr>
    </thead>
    <tbody>
{table_rows}    </tbody>
  </table>
</section>

<section id="species-explainability" style="margin-top:40px">
  <h2>Per-Species Explainability</h2>
  <p style="color:#555">Gradient-based k-mer importance and KG gene attention weights per species.</p>
  {"".join([
      f'<div style="margin:20px 0">'
      f'<h3>{SPECIES_NICE[sp]}</h3>'
      f'<a href="species/{sp}/figures/kmer_importance.html" target="_blank">'
      f'K-mer Importance</a> &nbsp;|&nbsp; '
      f'<a href="species/{sp}/figures/gene_attention.html" target="_blank">'
      f'Gene Attention</a> &nbsp;|&nbsp; '
      f'<a href="species/{sp}/figures/gate_distribution.html" target="_blank">'
      f'Fusion Gate</a>'
      f'</div>'
      for sp in SPECIES_LIST
      if os.path.exists(os.path.join(EXPLAIN_DIR, "species", sp, "figures",
                                     "kmer_importance.html"))
  ])}
</section>
"""

if os.path.exists(report_path):
    with open(report_path, "r") as f:
        html = f.read()

    # Inject before </body>
    if "</body>" in html:
        html = html.replace("</body>",
                            species_section_html + "\n</body>")
    else:
        html += species_section_html

    with open(report_path, "w") as f:
        f.write(html)
    print(f"Updated: {report_path}")
else:
    # Create a fresh minimal report
    fresh_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>KG-Trace — Results Dashboard</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:40px auto;padding:20px}}
h1{{color:#2c5282}} h2{{color:#2d3748;border-bottom:2px solid #e2e8f0;padding-bottom:6px}}
</style></head>
<body>
<h1>KG-Trace — Multi-Species Results Dashboard</h1>
{species_section_html}
</body></html>"""
    with open(report_path, "w") as f:
        f.write(fresh_html)
    print(f"Created: {report_path}")

print("\nStage M8 complete.")
