"""
Task 2: Attention vs SHAP scatter plot + export all figures as publication quality.
All values from actual computed outputs.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "KG-Trace" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
FIGURES_DIR = os.path.join(EXPLAIN_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── 1. Load alignment metrics ───────────────────────────────────────────────
print("[1/4] Loading alignment metrics...")
with open(os.path.join(EXPLAIN_DIR, "alignment_metrics.json")) as f:
    alignment = json.load(f)

gene_table = alignment["gene_importance_table"]
spearman_attn_shap = alignment.get("spearman_attn_vs_shap", {})
rho = spearman_attn_shap.get("rho", None)
pval = spearman_attn_shap.get("pvalue", None)

print(f"  {len(gene_table)} genes")
print(f"  Spearman rho = {rho}, p = {pval}")

# Load CARD gene set
with open(os.path.join(PROJECT_DIR, "kg/gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
card_gene_set = set(gene_mechanism.keys())

# ── 2. Build Attention vs SHAP scatter ───────────────────────────────────────
print("\n[2/4] Building scatter plot...")

gene_names = [g["gene"] for g in gene_table]
attn_ranks = [g["attn_rank"] for g in gene_table]
shap_ranks = [g["shap_rank"] for g in gene_table]
card_validated = [g["gene"] in card_gene_set for g in gene_table]
card_resistance_edges = [g.get("card_resistance_edges", 0) for g in gene_table]

colors = ["#2ecc71" if cv else "#e74c3c" for cv in card_validated]
sizes = [max(8, min(25, 8 + e * 2)) for e in card_resistance_edges]

fig_scatter = go.Figure()

# CARD-validated genes
card_idx = [i for i, cv in enumerate(card_validated) if cv]
non_card_idx = [i for i, cv in enumerate(card_validated) if not cv]

fig_scatter.add_trace(go.Scatter(
    x=[attn_ranks[i] for i in card_idx],
    y=[shap_ranks[i] for i in card_idx],
    mode="markers+text",
    marker=dict(size=[sizes[i] for i in card_idx], color="#2ecc71", 
                line=dict(width=1, color="white")),
    text=[gene_names[i] for i in card_idx],
    textposition="top center",
    textfont=dict(size=10, color="white"),
    name="CARD Validated",
    hovertemplate="<b>%{text}</b><br>Attention Rank: %{x}<br>SHAP Rank: %{y}<extra></extra>",
))

if non_card_idx:
    fig_scatter.add_trace(go.Scatter(
        x=[attn_ranks[i] for i in non_card_idx],
        y=[shap_ranks[i] for i in non_card_idx],
        mode="markers+text",
        marker=dict(size=[sizes[i] for i in non_card_idx], color="#e74c3c",
                    line=dict(width=1, color="white")),
        text=[gene_names[i] for i in non_card_idx],
        textposition="top center",
        textfont=dict(size=10, color="white"),
        name="Not CARD Validated",
        hovertemplate="<b>%{text}</b><br>Attention Rank: %{x}<br>SHAP Rank: %{y}<extra></extra>",
    ))

# Add diagonal line (perfect agreement)
fig_scatter.add_trace(go.Scatter(
    x=[1, 26], y=[1, 26],
    mode="lines",
    line=dict(dash="dash", color="rgba(255,255,255,0.3)"),
    showlegend=False,
))

rho_str = f"{rho:.4f}" if rho is not None else "n/a"
pval_str = f"{pval:.6f}" if pval is not None else "n/a"

fig_scatter.update_layout(
    title=f"Intrinsic vs Post-hoc Explanation Disagreement (ρ = {rho_str}, p = {pval_str})",
    xaxis_title="Gene Rank by Attention Weight (KG Encoder)",
    yaxis_title="Gene Rank by SHAP Importance (Genomic Encoder)",
    paper_bgcolor="#0a0a1a",
    plot_bgcolor="#1a1a2e",
    font=dict(color="white"),
    height=600, width=700,
    xaxis=dict(range=[0, 27], dtick=5, gridcolor="rgba(255,255,255,0.1)"),
    yaxis=dict(range=[0, 27], dtick=5, gridcolor="rgba(255,255,255,0.1)"),
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.5)"),
)

fig_scatter.add_annotation(
    text="Negative ρ = attention and SHAP capture<br>different biological signals",
    xref="paper", yref="paper",
    x=0.98, y=0.02, showarrow=False,
    bgcolor="rgba(255,200,0,0.2)", bordercolor="orange",
    font=dict(size=11, color="orange"),
    xanchor="right", yanchor="bottom",
)

# Save
fig_scatter.write_html(os.path.join(FIGURES_DIR, "attn_vs_shap_scatter.html"))
print(f"  Saved scatter HTML to {FIGURES_DIR}/attn_vs_shap_scatter.html")

# Try to save PNG/PDF (requires kaleido)
try:
    fig_scatter.write_image(os.path.join(FIGURES_DIR, "attn_vs_shap_scatter.png"), scale=3)
    fig_scatter.write_image(os.path.join(FIGURES_DIR, "attn_vs_shap_scatter.pdf"))
    print(f"  Saved scatter PNG + PDF")
except Exception as e:
    print(f"  WARNING: Could not save PNG/PDF (kaleido may not be installed): {e}")
    print(f"  HTML version saved successfully — use browser to export as image")

# ── 3. Export all dashboard figures ──────────────────────────────────────────
print("\n[3/4] Exporting all dashboard figures...")

# Re-create figures from dashboard data
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
gene_names_all = list(test_data["gene_names"])
attn_weights = test_data["attn_weights"]
gate_values = test_data["gate_values"]
gate_means = gate_values.mean(axis=1)

# SHAP data
shap_data = np.load(os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"), allow_pickle=True)
shap_values = shap_data["shap_values"]
feature_names = list(shap_data["feature_names"])

# Model results
with open(os.path.join(MODEL_DIR, "test_results.json")) as f:
    kg_results = json.load(f)
with open(os.path.join(os.path.join(PROJECT_DIR, "baselines"), "baseline_results.json")) as f:
    baseline_results = json.load(f)

n_test = len(test_data["test_ids"])

# Figure A: Gene Attention Heatmap
mean_attn = attn_weights.mean(axis=0)
top_gene_idx = np.argsort(mean_attn)[::-1][:20]
top_gene_n = [gene_names_all[i] for i in top_gene_idx]
rng = np.random.RandomState(42)
sample_idx = rng.choice(n_test, 100, replace=False)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=attn_weights[sample_idx][:, top_gene_idx].T,
    x=[f"G{i}" for i in range(100)],
    y=top_gene_n,
    colorscale="Viridis",
))
fig_heatmap.update_layout(
    title="Gene Attention Heatmap — Top 20 Genes × 100 Genomes",
    xaxis_title="Test Genomes", yaxis_title="Resistance Genes",
    height=500, width=900,
    font=dict(color="black"),
    margin=dict(l=120),
)
fig_heatmap.write_html(os.path.join(FIGURES_DIR, "gene_attention_heatmap.html"))
try:
    fig_heatmap.write_image(os.path.join(FIGURES_DIR, "gene_attention_heatmap.png"), scale=3)
    fig_heatmap.write_image(os.path.join(FIGURES_DIR, "gene_attention_heatmap.pdf"))
except Exception:
    pass
print("  Saved gene_attention_heatmap")

# Figure B: Gate Distribution
fig_gate = go.Figure()
fig_gate.add_trace(go.Histogram(x=gate_means, nbinsx=50, marker_color="#00d2ff"))
fig_gate.add_vline(x=0.5, line_dash="dash", line_color="red")
fig_gate.update_layout(
    title=f"Fusion Gate Distribution (mean={gate_means.mean():.4f})",
    xaxis_title="Gate Mean Value", yaxis_title="Count",
    height=400, width=700,
)
fig_gate.write_html(os.path.join(FIGURES_DIR, "fusion_gate_distribution.html"))
try:
    fig_gate.write_image(os.path.join(FIGURES_DIR, "fusion_gate_distribution.png"), scale=3)
    fig_gate.write_image(os.path.join(FIGURES_DIR, "fusion_gate_distribution.pdf"))
except Exception:
    pass
print("  Saved fusion_gate_distribution")

# Figure C: SHAP Top-30 Bar
shap_mean_abs = np.abs(shap_values).mean(axis=0)
top_shap_idx = np.argsort(shap_mean_abs)[::-1][:30]
top_shap_n = [feature_names[i] for i in top_shap_idx]
top_shap_imp = shap_mean_abs[top_shap_idx]
shap_colors = ["#2ecc71" if n.split(":")[0] in card_gene_set else "#e74c3c" for n in top_shap_n]

fig_shap = go.Figure()
fig_shap.add_trace(go.Bar(
    x=top_shap_imp[::-1], y=top_shap_n[::-1], orientation="h",
    marker=dict(color=shap_colors[::-1]),
    text=[f"{v:.4f}" for v in top_shap_imp[::-1]], textposition="outside",
))
fig_shap.update_layout(
    title="Top 30 SHAP Features (gradient × input)",
    xaxis_title="Mean |SHAP|",
    height=800, width=800, margin=dict(l=200),
)
fig_shap.write_html(os.path.join(FIGURES_DIR, "shap_top30_bar.html"))
try:
    fig_shap.write_image(os.path.join(FIGURES_DIR, "shap_top30_bar.png"), scale=3)
    fig_shap.write_image(os.path.join(FIGURES_DIR, "shap_top30_bar.pdf"))
except Exception:
    pass
print("  Saved shap_top30_bar")

# Figure D: Confusion matrices
all_models_cm = [("KG-Trace", kg_results["confusion_matrix"])]
for br in baseline_results:
    all_models_cm.append((br["model"], br["confusion_matrix"]))

for model_name, cm in all_models_cm:
    cm_arr = np.array(cm)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_arr, x=["Pred S", "Pred R"], y=["True S", "True R"],
        text=cm_arr, texttemplate="%{text}", textfont=dict(size=16),
        colorscale="Blues", showscale=False,
    ))
    safe_name = model_name.replace(" ", "_").replace("-", "")
    fig_cm.update_layout(
        title=f"Confusion Matrix — {model_name}",
        height=350, width=400,
    )
    fig_cm.write_html(os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.html"))
    try:
        fig_cm.write_image(os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.png"), scale=3)
        fig_cm.write_image(os.path.join(FIGURES_DIR, f"confusion_matrix_{safe_name}.pdf"))
    except Exception:
        pass
    print(f"  Saved confusion_matrix_{safe_name}")

# ── 4. List all exported figures ─────────────────────────────────────────────
print(f"\n[4/4] All figures in {FIGURES_DIR}/:")
for f in sorted(os.listdir(FIGURES_DIR)):
    size = os.path.getsize(os.path.join(FIGURES_DIR, f))
    print(f"  {f:50s} {size/1024:.0f} KB")

print("DONE — generate_figures.py")
