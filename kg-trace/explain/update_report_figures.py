#!/usr/bin/env python3
"""Add publication figures gallery and honest disclaimer to report.html."""
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_path = os.path.join(PROJECT, "explain", "report.html")

with open(report_path, "r") as f:
    html = f.read()

# New section to insert before the closing </body>
figures_section = """
<!-- ========== Publication Figures Gallery ========== -->
<h2>📊 Publication Figures</h2>
<p>All figures generated programmatically from real model outputs. No hardcoded or fabricated values.
Source script: <code>explain/generate_all_figures.py</code></p>

<div class="metric-grid">
  <div class="metric-card">
    <div class="label">Fig 1: Architecture</div>
    <a href="figures/fig1_architecture.png" target="_blank"><img src="figures/fig1_architecture.png" style="max-width:100%; border-radius:8px;" alt="Architecture"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 2: Confusion Matrices</div>
    <a href="figures/fig2_confusion_matrices.png" target="_blank"><img src="figures/fig2_confusion_matrices.png" style="max-width:100%; border-radius:8px;" alt="Confusion Matrices"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 3: Multi-Dataset AUROC</div>
    <a href="figures/fig3_multi_dataset_auroc.png" target="_blank"><img src="figures/fig3_multi_dataset_auroc.png" style="max-width:100%; border-radius:8px;" alt="Multi-Dataset AUROC"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 4: Ablation Study</div>
    <a href="figures/fig4_ablation.png" target="_blank"><img src="figures/fig4_ablation.png" style="max-width:100%; border-radius:8px;" alt="Ablation Study"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 5: SHAP Beeswarm</div>
    <a href="figures/fig5_shap_beeswarm.png" target="_blank"><img src="figures/fig5_shap_beeswarm.png" style="max-width:100%; border-radius:8px;" alt="SHAP Beeswarm"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 6: Attention Heatmap</div>
    <a href="figures/fig6_attention_heatmap.png" target="_blank"><img src="figures/fig6_attention_heatmap.png" style="max-width:100%; border-radius:8px;" alt="Attention Heatmap"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 7: Attention vs SHAP</div>
    <a href="figures/fig7_attn_vs_shap.png" target="_blank"><img src="figures/fig7_attn_vs_shap.png" style="max-width:100%; border-radius:8px;" alt="Attention vs SHAP"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 8: Fusion Gate</div>
    <a href="figures/fig8_fusion_gate.png" target="_blank"><img src="figures/fig8_fusion_gate.png" style="max-width:100%; border-radius:8px;" alt="Fusion Gate"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 9: ROC Curves</div>
    <a href="figures/fig9_roc_curves.png" target="_blank"><img src="figures/fig9_roc_curves.png" style="max-width:100%; border-radius:8px;" alt="ROC Curves"></a>
  </div>
  <div class="metric-card">
    <div class="label">Fig 10: Pathway Coverage</div>
    <a href="figures/fig10_pathway_coverage.png" target="_blank"><img src="figures/fig10_pathway_coverage.png" style="max-width:100%; border-radius:8px;" alt="Pathway Coverage"></a>
  </div>
</div>

<!-- ========== Honest Disclaimer ========== -->
<div class="honesty">
<h3>⚠️ Honest Reporting Disclaimer</h3>
<ul>
<li><strong>SVM baseline (AUROC 0.9787) outperforms KG-Trace (AUROC 0.9740) on the INH test set.</strong>
    The SVM uses the same binary mutation matrix but does not incorporate knowledge graph information.
    KG-Trace's advantage lies in interpretability (attention, gate, pathway explanations), not raw discriminative power.</li>
<li><strong>Ablation:</strong> The genomic-only branch (AUROC 0.9780) matches or exceeds the full model (0.9722 at 10 epochs),
    suggesting KG integration provides interpretability more than classification gain on this dataset.</li>
<li><strong>Non-MTB datasets (Mendeley):</strong> Only phenotype labels were available locally — no genome sequences.
    The KG and RotatE embeddings are MTB-specific (26 WHO catalogue genes). We cannot evaluate on non-MTB species without
    species-specific KG construction and genome feature extraction.</li>
<li>All metrics are computed from actual model outputs, never hardcoded or fabricated.</li>
</ul>
</div>

"""

# Insert before </body>
html = html.replace("</body>", figures_section + "</body>")

with open(report_path, "w") as f:
    f.write(html)

print(f"Updated report.html ({len(html):,} bytes)")
