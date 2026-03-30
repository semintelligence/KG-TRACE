#!/usr/bin/env python3
"""
Generate all 10 publication figures for KG-AMR v2.
Reads only from real result files — no hardcoded metrics.
Saves each figure as .html + .png + .pdf in explain/figures/.
"""
import os, json, sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix

PROJECT = os.path.expanduser("~/Desktop/AMR NamanXSarika/kg-amr-v2")
os.chdir(PROJECT)
FIG_DIR = os.path.join(PROJECT, "explain", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCALE = 3  # PNG resolution multiplier


def save_fig(fig, name):
    """Save figure as HTML, PNG, and PDF."""
    html_path = os.path.join(FIG_DIR, f"{name}.html")
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.write_html(html_path)
    fig.write_image(png_path, scale=SCALE)
    fig.write_image(pdf_path)
    print(f"  Saved: {name}.html / .png / .pdf")


# ============================================================
# Figure 1: Architecture Diagram (matplotlib)
# ============================================================
def fig1_architecture():
    """Architecture block diagram using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("KG-AMR v2 Architecture", fontsize=16, fontweight="bold", pad=20)

    def box(x, y, w, h, text, color="#4A90D9", fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                        facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def arrow(x1, y1, x2, y2, text="", color="black"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        if text:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.1, my+0.15, text, fontsize=7, color=color)

    # Input boxes
    box(0.5, 7.5, 3, 1.2, "Binary Mutation\nMatrix\n(41,460 × 17,352)", "#2C3E50", 8)
    box(0.5, 5.5, 3, 1.2, "KG Gene Embeddings\n(RotatE, 26 × 64)", "#8E44AD", 8)

    # Encoder blocks
    box(5, 7.5, 3, 1.2, "Genomic Encoder\nLinear→BN→ReLU→Drop\n→Linear (→256d)", "#2980B9", 8)
    box(5, 5.5, 3, 1.2, "KG Encoder\nSelf-Attention Pooling\n(→64d)", "#8E44AD", 8)

    # Projections
    box(5, 3.5, 3, 0.8, "Projection\n(256→128d / 64→128d)", "#27AE60", 8)

    # Fusion
    box(5, 2, 3, 0.8, "Cross-Attention Gate\nσ(MLP([g;k]))", "#E67E22", 8)

    # Fusion output
    box(5, 0.5, 3, 0.8, "Fused Representation\n(128d)", "#E74C3C", 8)

    # Output heads
    box(10, 7.5, 3, 1, "AMR Classifier\nLinear→ReLU→Linear(2)\n(R / S)", "#C0392B", 8)
    box(10, 5.5, 3, 1, "Gene Detection Head\nLinear→ReLU→Linear(26)\n→Sigmoid", "#16A085", 8)

    # KG box
    box(10, 3.5, 3, 1, "Knowledge Graph\n60,017 triples\n25,095 entities", "#7F8C8D", 8)

    # Arrows: inputs to encoders
    arrow(3.5, 8.1, 5, 8.1)
    arrow(3.5, 6.1, 5, 6.1)

    # Encoders to projection
    arrow(6.5, 7.5, 6.5, 4.3)

    # Projection to fusion
    arrow(6.5, 3.5, 6.5, 2.8)

    # Fusion to fused
    arrow(6.5, 2, 6.5, 1.3)

    # Fused to output heads
    arrow(8, 0.9, 11.5, 5.5, "")
    arrow(8, 0.9, 11.5, 7.5, "")

    # KG to KG encoder
    arrow(10, 4.0, 8, 6.1, "RotatE")

    plt.tight_layout()
    png_path = os.path.join(FIG_DIR, "fig1_architecture.png")
    pdf_path = os.path.join(FIG_DIR, "fig1_architecture.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig1_architecture.png / .pdf")


# ============================================================
# Figure 2: Confusion Matrices (2×2 grid)
# ============================================================
def fig2_confusion_matrices():
    """2×2 grid: KG-AMR v2, SVM, XGBoost, RF confusion matrices."""
    # Load KG-AMR outputs
    to = np.load("model/test_outputs.npz", allow_pickle=True)
    bp = np.load("baselines/baseline_predictions.npz", allow_pickle=True)

    models = {
        "KG-AMR v2": (to["preds"], to["labels"]),
        "SVM": (bp["svm_preds"], bp["y_test"]),
        "XGBoost": (bp["xgb_preds"], bp["y_test"]),
        "Random Forest": (bp["rf_preds"], bp["y_test"]),
    }

    fig = make_subplots(rows=2, cols=2, subplot_titles=list(models.keys()),
                        horizontal_spacing=0.12, vertical_spacing=0.12)
    labels_text = ["Susceptible", "Resistant"]

    for idx, (name, (preds, labels)) in enumerate(models.items()):
        row, col = divmod(idx, 2)
        cm = confusion_matrix(labels, preds)
        # Normalize for display
        cm_pct = cm / cm.sum() * 100

        text_vals = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]

        heatmap = go.Heatmap(
            z=cm, x=labels_text, y=labels_text,
            text=text_vals, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"
        )
        fig.add_trace(heatmap, row=row+1, col=col+1)
        fig.update_xaxes(title_text="Predicted", row=row+1, col=col+1)
        fig.update_yaxes(title_text="True", row=row+1, col=col+1)

    fig.update_layout(title="Confusion Matrices — INH Test Set (n=5,665)",
                      height=700, width=800, template="plotly_white")
    save_fig(fig, "fig2_confusion_matrices")


# ============================================================
# Figure 3: Multi-Dataset AUROC Bar Chart
# ============================================================
def fig3_multi_dataset_auroc():
    """Bar chart of AUROC across multiple datasets."""
    df = pd.read_csv("evaluate/multi_dataset_results.csv")
    # Filter to rows with actual AUROC values
    df_valid = df.dropna(subset=["AUROC"]).copy()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_valid["Dataset"],
        y=df_valid["AUROC"],
        marker_color=["#2980B9", "#27AE60", "#E67E22", "#E74C3C"][:len(df_valid)],
        text=df_valid["AUROC"].round(4),
        textposition="outside",
        hovertemplate="Dataset: %{x}<br>AUROC: %{y:.4f}<br>N_test: %{customdata}<extra></extra>",
        customdata=df_valid["N_test"].astype(int),
    ))
    fig.update_layout(
        title="KG-AMR v2 AUROC Across Drug Datasets",
        xaxis_title="Dataset",
        yaxis_title="AUROC",
        yaxis=dict(range=[0.9, 1.0]),
        template="plotly_white",
        height=500, width=700,
    )
    save_fig(fig, "fig3_multi_dataset_auroc")


# ============================================================
# Figure 4: Ablation Study
# ============================================================
def fig4_ablation():
    """Grouped bar chart for ablation study (AUROC + F1)."""
    df = pd.read_csv("evaluate/ablation_results.csv")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AUROC", x=df["Config"], y=df["AUROC"],
        marker_color="#2980B9",
        text=df["AUROC"].round(4), textposition="outside"
    ))
    # Handle column name (might be "F1-macro" or "F1_macro")
    f1_col = "F1-macro" if "F1-macro" in df.columns else "F1_macro"
    fig.add_trace(go.Bar(
        name="F1-macro", x=df["Config"], y=df[f1_col],
        marker_color="#E67E22",
        text=df[f1_col].round(4), textposition="outside"
    ))
    fig.update_layout(
        title="Ablation Study — 10-Epoch Comparison",
        barmode="group",
        yaxis=dict(range=[0.8, 1.0]),
        template="plotly_white",
        height=500, width=900,
    )
    save_fig(fig, "fig4_ablation")


# ============================================================
# Figure 5: SHAP Beeswarm (Top 20 Features)
# ============================================================
def fig5_shap_beeswarm():
    """Beeswarm-style plot of top 20 SHAP features."""
    raw = np.load("explain/shap_raw_values.npz", allow_pickle=True)
    shap_vals = raw["shap_values"]       # (5665, 17352)
    feat_names = raw["feature_names"]    # (17352,)

    # Mean absolute SHAP per feature
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[-20:][::-1]

    # For beeswarm: sample up to 500 random genomes for visibility
    rng = np.random.RandomState(42)
    n_samples = min(500, shap_vals.shape[0])
    sample_idx = rng.choice(shap_vals.shape[0], n_samples, replace=False)

    fig = go.Figure()
    for rank, fi in enumerate(top_idx):
        vals = shap_vals[sample_idx, fi]
        jitter = rng.normal(0, 0.15, size=len(vals))
        fig.add_trace(go.Scatter(
            x=vals, y=[rank]*len(vals) + jitter,
            mode="markers",
            marker=dict(size=3, color=vals, colorscale="RdBu_r",
                        cmin=-np.percentile(np.abs(shap_vals), 99),
                        cmax=np.percentile(np.abs(shap_vals), 99),
                        opacity=0.6),
            name=feat_names[fi],
            hovertemplate=f"{feat_names[fi]}<br>SHAP: %{{x:.6f}}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        title="SHAP Feature Importance (Top 20, Beeswarm)",
        xaxis_title="SHAP Value",
        yaxis=dict(tickvals=list(range(20)),
                   ticktext=[feat_names[i] for i in top_idx],
                   autorange="reversed"),
        template="plotly_white",
        height=700, width=900,
    )
    save_fig(fig, "fig5_shap_beeswarm")


# ============================================================
# Figure 6: Gene Attention Heatmap
# ============================================================
def fig6_attention_heatmap():
    """Heatmap of attention weights across genes (sampled genomes)."""
    to = np.load("model/test_outputs.npz", allow_pickle=True)
    attn = to["attn_weights"]       # (5665, 26)
    gene_names = to["gene_names"]   # (26,)
    labels = to["labels"]           # (5665,)

    # Sort by label then mean attention
    order = np.argsort(labels)
    attn_sorted = attn[order]

    # Subsample for visibility (max 200 rows)
    n_show = min(200, attn_sorted.shape[0])
    step = attn_sorted.shape[0] // n_show
    attn_sub = attn_sorted[::step][:n_show]
    labels_sub = labels[order][::step][:n_show]

    # Y-axis labels
    y_labels = [f"{'R' if l==1 else 'S'}" for l in labels_sub]

    fig = go.Figure(data=go.Heatmap(
        z=attn_sub,
        x=list(gene_names),
        y=y_labels,
        colorscale="YlOrRd",
        hovertemplate="Gene: %{x}<br>Genome: %{y}<br>Attn: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Gene Attention Weights (n={n_show} sampled genomes, sorted by label)",
        xaxis_title="Gene",
        yaxis_title="Genome (S=Susceptible, R=Resistant)",
        template="plotly_white",
        height=700, width=1000,
    )
    save_fig(fig, "fig6_attention_heatmap")


# ============================================================
# Figure 7: Attention vs SHAP Scatter
# ============================================================
def fig7_attn_vs_shap():
    """Scatter: attention rank vs SHAP rank per gene."""
    with open("explain/alignment_metrics.json") as f:
        am = json.load(f)
    git = am["gene_importance_table"]  # list of dicts

    df = pd.DataFrame(git)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["attn_rank"], y=df["shap_rank"],
        mode="markers+text",
        text=df["gene"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=df["card_resistance_edges"].clip(1).apply(np.log1p) * 5 + 5,
            color=df["card_rank"],
            colorscale="Viridis_r",
            colorbar=dict(title="CARD Rank"),
            opacity=0.7,
        ),
        hovertemplate=(
            "Gene: %{text}<br>Attn rank: %{x}<br>SHAP rank: %{y}<br>"
            "CARD edges: %{customdata}<extra></extra>"
        ),
        customdata=df["card_resistance_edges"],
    ))
    # Perfect agreement line
    max_rank = max(df["attn_rank"].max(), df["shap_rank"].max())
    fig.add_trace(go.Scatter(
        x=[1, max_rank], y=[1, max_rank],
        mode="lines", line=dict(dash="dash", color="gray"),
        showlegend=False,
    ))
    spearman = am.get("spearman_attn_vs_shap", {})
    rho = spearman.get("rho", "N/A") if isinstance(spearman, dict) else spearman
    fig.update_layout(
        title=f"Attention Rank vs SHAP Rank per Gene (Spearman ρ={rho})",
        xaxis_title="Attention Rank",
        yaxis_title="SHAP Rank",
        template="plotly_white",
        height=600, width=700,
    )
    save_fig(fig, "fig7_attn_vs_shap")


# ============================================================
# Figure 8: Fusion Gate Distribution
# ============================================================
def fig8_fusion_gate():
    """Histogram of per-genome mean gate values, colored by label."""
    df = pd.read_csv("explain/fusion_gate_values.csv")

    fig = go.Figure()
    for label, color, name in [(0, "#3498DB", "Susceptible"), (1, "#E74C3C", "Resistant")]:
        label_str = "SUSCEPTIBLE" if label == 0 else "RESISTANT"
        mask = df["true_label"] == label_str
        fig.add_trace(go.Histogram(
            x=df.loc[mask, "gate_mean"],
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=50,
        ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                  annotation_text="Equal weighting")
    fig.update_layout(
        title="Fusion Gate Distribution (gate<0.5 → KG-dominated, gate>0.5 → Genomic-dominated)",
        xaxis_title="Mean Gate Value",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=500, width=800,
    )
    save_fig(fig, "fig8_fusion_gate")


# ============================================================
# Figure 9: ROC Curves Overlay
# ============================================================
def fig9_roc_curves():
    """ROC curves for KG-AMR v2 and all baselines on same test set."""
    to = np.load("model/test_outputs.npz", allow_pickle=True)
    bp = np.load("baselines/baseline_predictions.npz", allow_pickle=True)

    labels = to["labels"]

    curves = {
        "KG-AMR v2": to["probs"],
        "SVM": bp["svm_probs"],
        "XGBoost": bp["xgb_probs"],
        "Random Forest": bp["rf_probs"],
    }
    colors = {"KG-AMR v2": "#E74C3C", "SVM": "#2980B9",
              "XGBoost": "#27AE60", "Random Forest": "#8E44AD"}

    fig = go.Figure()
    for name, probs in curves.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={roc_auc:.4f})",
            line=dict(color=colors[name], width=2),
        ))
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), showlegend=False,
    ))
    fig.update_layout(
        title="ROC Curves — INH Test Set",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white",
        height=600, width=700,
        legend=dict(x=0.55, y=0.05),
    )
    save_fig(fig, "fig9_roc_curves")


# ============================================================
# Figure 10: Pathway Coverage Summary
# ============================================================
def fig10_pathway_coverage():
    """Bar chart showing pathway reachability per gene."""
    with open("explain/pathway_explanations.json") as f:
        pe = json.load(f)

    gene_contribs = pe.get("gene_contributions", {})
    if not gene_contribs:
        print("  SKIPPED fig10: no gene_contributions data")
        return

    genes = sorted(gene_contribs.keys())
    n_paths = []
    n_resistance = []
    for g in genes:
        info = gene_contribs[g]
        if isinstance(info, dict):
            n_paths.append(info.get("n_paths_any", info.get("n_paths", 0)))
            n_resistance.append(info.get("n_resistance_paths", 0))
        else:
            n_paths.append(0)
            n_resistance.append(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=genes, y=n_paths, name="Any Path",
        marker_color="#3498DB",
    ))
    fig.add_trace(go.Bar(
        x=genes, y=n_resistance, name="Resistance Path",
        marker_color="#E74C3C",
    ))

    coverage_any = pe.get("pathway_coverage_any_pct", "N/A")
    coverage_r = pe.get("pathway_coverage_resistance_pct", "N/A")
    fig.update_layout(
        title=f"KG Pathway Reachability per Gene (coverage: any={coverage_any}%, resistance={coverage_r}%)",
        xaxis_title="Gene",
        yaxis_title="Number of Reachable Pathways",
        barmode="group",
        template="plotly_white",
        height=500, width=1000,
    )
    save_fig(fig, "fig10_pathway_coverage")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating publication figures...")

    print("\n[1/10] Architecture diagram")
    fig1_architecture()

    print("\n[2/10] Confusion matrices")
    fig2_confusion_matrices()

    print("\n[3/10] Multi-dataset AUROC")
    fig3_multi_dataset_auroc()

    print("\n[4/10] Ablation study")
    fig4_ablation()

    print("\n[5/10] SHAP beeswarm")
    fig5_shap_beeswarm()

    print("\n[6/10] Gene attention heatmap")
    fig6_attention_heatmap()

    print("\n[7/10] Attention vs SHAP scatter")
    fig7_attn_vs_shap()

    print("\n[8/10] Fusion gate distribution")
    fig8_fusion_gate()

    print("\n[9/10] ROC curves")
    fig9_roc_curves()

    print("\n[10/10] Pathway coverage")
    fig10_pathway_coverage()

    print("\n✅ All figures generated in explain/figures/")
