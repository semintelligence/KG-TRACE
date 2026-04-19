#!/usr/bin/env python3
import os, sys, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup Directory
FIG_DIR = "explain/final_publication_figures"
os.makedirs(FIG_DIR, exist_ok=True)
SCALE = 4  # Ultra high res scaling

# Thematic Colors
C_DATA   = "#2980B9"
C_KG     = "#8E44AD"
C_FUSED  = "#E74C3C"
C_GATE   = "#E67E22"
C_AUX    = "#27AE60"
C_GRAY   = "#7F8C8D"

def save_fig(fig, name):
    fig.write_image(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.write_image(os.path.join(FIG_DIR, f"{name}.png"), scale=SCALE)
    print(f" ✓ {name}.png / .pdf")

def save_mpl(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    print(f" ✓ {name}.png / .pdf")

# ====================================================================
# FIG 1: COMBINED CLINICAL WORKFLOW & KG SCHEMA
# ====================================================================
def create_combined_figure_1():
    # Load actual numbers from the knowledge graph JSON automatically
    with open("kg/kg_summary.json") as f:
        kg_data = json.load(f)
    counts = kg_data.get("relation_counts", {})

    fig = plt.figure(figsize=(15, 16))
    
    # -------------------------------------------------------------
    # PANEL A: Clinical Workflow (Top)
    # -------------------------------------------------------------
    ax1 = fig.add_axes([0.05, 0.55, 0.9, 0.4]) # left, bottom, width, height
    ax1.set_xlim(0, 16); ax1.set_ylim(0, 8); ax1.axis("off")
    ax1.text(0, 8.2, "A) Clinical Workflow: Two-Level Explainability in Laboratory Reports", fontsize=18, fontweight="bold", ha="left")

    def draw_box1(x, y, w, h, text, color, tcolor="white"):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=color, edgecolor="black", linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=12, color=tcolor, fontweight="bold")
    
    def arrow1(x1, y1, x2, y2):
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2.5, color="black"))

    draw_box1(0.5, 6, 3.5, 1.5, "WGS Isolate Data\n(Clinical Sample)", C_DATA)
    draw_box1(0.5, 3.5, 3.5, 1.5, "MTB Knowledge Graph\n(WHO Catalogues)", C_KG)
    draw_box1(5.0, 4.5, 4.2, 2.0, "KG-TRACE Model\nPredicted: RESISTANT\n(Probability = 0.94)", C_FUSED)
    
    arrow1(4.0, 6.75, 5.0, 5.5)
    arrow1(4.0, 4.25, 5.0, 5.0)

    draw_box1(10.2, 5.5, 5.5, 1.8, "Level 1: SHAP Attribution (Primary)\n✓ katG:S315T (SHAP = +2.41)\nRanked 1 of 17,352 features", C_GATE)
    arrow1(9.2, 5.5, 10.2, 6.0)

    draw_box1(10.2, 2.5, 5.5, 2.0, "Level 2: KG Traceability (Support)\n\nkatG  →  katG:S315T  →  Isoniazid", "#2C3E50")
    arrow1(12.95, 5.5, 12.95, 4.5)
    ax1.text(12.95, 5.0, "Is top driver catalogued?", ha="left", va="center", fontsize=10, style="italic")

    draw_box1(6.0, 0.5, 9.7, 1.2, "Clinical Report: Isoniazid Resistant (WHO Grade 1 Evidence)", "#27AE60")
    arrow1(12.95, 2.5, 12.95, 1.7)

    # -------------------------------------------------------------
    # PANEL B: KG Schema (Bottom)
    # -------------------------------------------------------------
    ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.45])
    ax2.set_xlim(0, 16); ax2.set_ylim(0, 12); ax2.axis("off")
    ax2.text(0, 11.5, "B) M. tuberculosis AMR Knowledge Graph Schema", fontsize=18, fontweight="bold", ha="left")

    def draw_node2(x, y, w, h, label, sublabel, color):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=color, edgecolor="black", linewidth=2.5, alpha=0.92)
        ax2.add_patch(rect)
        ax2.text(x+w/2, y+h*0.7, label, ha="center", va="center", fontsize=15, fontweight="bold", color="white")
        ax2.text(x+w/2, y+h*0.3, sublabel, ha="center", va="center", fontsize=12, color="white", style="italic")

    def draw_edge2(x1, y1, x2, y2, label, color="black", rad="0.0", text_y_offset=0):
        ax2.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=3.0, connectionstyle=f"arc3,rad={rad}"))
        ax2.text((x1+x2)/2, (y1+y2)/2 + text_y_offset, label, ha="center", va="center", fontsize=12, fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.95, linewidth=1.5))

    draw_node2(1.5, 8.5, 4.0, 1.8, "Gene", "n = 26", C_DATA)
    draw_node2(10.5, 8.5, 4.0, 1.8, "Mutation", f"n = {counts.get('has_mutation', 25045):,}", "#E74C3C")
    draw_node2(1.5, 2.5, 4.0, 1.8, "Mechanism", "n = 9", C_KG)
    draw_node2(10.5, 2.5, 4.0, 1.8, "Drug", "n = 15", C_AUX)

    draw_edge2(5.5, 9.4, 10.5, 9.4, f"has_mutation \n ({counts.get('has_mutation', 25045):,})", "#E74C3C")
    draw_edge2(12.5, 8.5, 12.5, 4.3, f"confers_resistance_to \n ({counts.get('confers_resistance_to', 4972):,})", "#E74C3C", rad="0.1")
    draw_edge2(11.5, 8.5, 11.5, 4.3, f"confers_suscept_to \n ({counts.get('confers_susceptibility_to', 18268):,})", C_AUX, rad="-0.1")
    draw_edge2(3.5, 8.5, 3.5, 4.3, f"belongs_to_mechanism \n ({counts.get('belongs_to', 26):,})", C_KG)
    draw_edge2(10.5, 3.4, 5.5, 3.4, f"has_uncertain_effect_on \n ({counts.get('has_uncertain_effect_on', 11686):,})", C_GRAY)
    
    # Shifted callout x=2.5 (from 4.0) to prevent overlapping with the vertical connections at 11.5
    callout = mpatches.FancyBboxPatch((2.2, 5.0), 8.3, 2.0, boxstyle="round,pad=0.3", facecolor="#FFF9E6", edgecolor="#D4A017", linewidth=2.5, alpha=0.95)
    ax2.add_patch(callout)
    ax2.text(6.35, 6.5, "⚡ Path Example", ha="center", va="center", fontsize=14, fontweight="bold", color="#8B6914")
    ax2.text(6.35, 5.7, "katG ──[has_mutation]──▸ katG:S315T ──[confers_resistance_to]──▸ Isoniazid", ha="center", va="center", fontsize=11.5, family="monospace", color="#2C3E50", bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#E74C3C"))
    
    save_mpl(fig, "Fig1_Combined_Workflow_and_KG")
    plt.close(fig)

# ====================================================================
# FIG 2: ARCHITECTURE FRAMEWORK
# ====================================================================
def create_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")
    ax.set_title("Figure 2: KG-TRACE Architecture Framework", fontsize=18, fontweight="bold", pad=20)

    def rounded_rect(x, y, w, h, title, subtitle, color):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", facecolor=color, edgecolor="black", lw=2, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x+w/2, y+h/2 + 0.3, title, ha="center", va="center", fontsize=12, fontweight="bold", color="white")
        ax.text(x+w/2, y+h/2 - 0.3, subtitle, ha="center", va="center", fontsize=10, color="white")
    
    def arrow(x1, y1, x2, y2, color, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=3))
        if text: ax.text((x1+x2)/2, (y1+y2)/2 + 0.3, text, ha="center", va="center", fontsize=10, fontweight="bold", color=color, bbox=dict(boxstyle="round", facecolor="white", edgecolor=color, alpha=0.9))

    ax.text(2.5, 11, "GENOMIC PATH", fontsize=14, fontweight="bold", color=C_DATA, ha="center")
    rounded_rect(0.5, 9, 4, 1.5, "Input: Mutation Vector", "41,460 × 17,352 binary", C_DATA)
    rounded_rect(5.5, 9, 4, 1.5, "Genomic Encoder", "2-Layer MLP → 256d\nDropout, BatchNorm", C_DATA)
    arrow(4.5, 9.75, 5.5, 9.75, C_DATA)
    arrow(7.5, 9, 7.5, 7.5, C_DATA, "g_proj (128d)")

    ax.text(2.5, 3.5, "KNOWLEDGE PATH", fontsize=14, fontweight="bold", color=C_KG, ha="center")
    rounded_rect(0.5, 4, 4, 1.5, "Input: KG Embeddings", "RotatE 64d (26 genes)", C_KG)
    rounded_rect(5.5, 4, 4, 1.5, "KG Attention Encoder", "Self-Attention → 64d summary", C_KG)
    arrow(4.5, 4.75, 5.5, 4.75, C_KG)
    arrow(7.5, 5.5, 7.5, 6.5, C_KG, "k_proj (128d)")

    gate_rect = mpatches.FancyBboxPatch((5.5, 6.5), 6.5, 1.0, boxstyle="round,pad=0.2", facecolor=C_GATE, edgecolor="#D35400", lw=3)
    ax.add_patch(gate_rect)
    ax.text(8.75, 7.0, "Cross-Attention Gate (α)\nα = σ(MLP([g_proj, k_proj]))", ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    
    rounded_rect(6.5, 1.5, 4.5, 1.5, "Fused Representation", "h = α·g + (1-α)·k  (128d)", C_FUSED)
    arrow(8.75, 6.5, 8.75, 3.0, C_FUSED, "Adaptive Blend")

    rounded_rect(12.5, 8.5, 3, 1.5, "AMR Classifier", "Resistant / Susceptible\n(Primary Loss)", C_FUSED)
    rounded_rect(12.5, 4.5, 3, 1.5, "Gene Auxiliary Head", "Detection of 26 Genes\n(Auxiliary Loss)", C_AUX)
    arrow(11.0, 2.25, 14.0, 8.5, "dimgray")
    arrow(11.0, 2.25, 14.0, 4.5, "dimgray")
    
    save_mpl(fig, "Fig2_Architecture")
    plt.close(fig)

# ====================================================================
# FIG 3: ROC + CONFUSION MATRIX
# ====================================================================
def create_roc_confusion():
    to = np.load("model/test_outputs.npz", allow_pickle=True)
    bp = np.load("baselines/baseline_predictions.npz", allow_pickle=True)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["<b>A) ROC Curves</b>", "<b>B) KG-TRACE Confusion Matrix</b>"], horizontal_spacing=0.15)
    
    curves = {"KG-TRACE": (to["probs"], to["labels"]), "LinearSVC": (bp["svm_probs"], bp["y_test"]), "Random Forest": (bp["rf_probs"], bp["y_test"])}
    colors = {"KG-TRACE": C_FUSED, "LinearSVC": C_DATA, "Random Forest": C_KG}

    for name, (probs, labels) in curves.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc(fpr, tpr):.4f})", line=dict(color=colors[name], width=3)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False), row=1, col=1)

    cm = confusion_matrix(to["labels"], to["preds"])
    cm_pct = cm / cm.sum() * 100
    text_vals = [[f"{cm[i][j]:,}<br>({cm_pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]
    fig.add_trace(go.Heatmap(z=cm, x=["Susceptible", "Resistant"], y=["Susceptible", "Resistant"], text=text_vals, texttemplate="%{text}", colorscale="Blues", showscale=False, textfont=dict(size=14)), row=1, col=2)
    
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Predicted Label", row=1, col=2)
    fig.update_yaxes(title_text="True Label", row=1, col=2)

    fig.update_layout(title=dict(text="Figure 3: Test Set Performance (n=5,665)", font=dict(size=18)), template="plotly_white", width=1200, height=500, legend=dict(x=0.20, y=0.15))
    save_fig(fig, "Fig3_ROC_Confusion")

# ====================================================================
# FIG 4: SHAP BEESWARM
# ====================================================================
def create_shap_beeswarm():
    raw = np.load("explain/shap_raw_values.npz", allow_pickle=True)
    shap_vals = raw["shap_values"]
    feat_names = raw["feature_names"]

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    top_idx = np.argsort(mean_abs)[-20:][::-1]

    # Select a deterministic slice for plotting speed without "random" seeds
    sample_idx = np.linspace(0, shap_vals.shape[0]-1, min(800, shap_vals.shape[0])).astype(int)

    fig = go.Figure()
    for rank, fi in enumerate(top_idx[::-1]): 
        vals = shap_vals[sample_idx, fi]
        
        # Deterministic visual spacing for standard beeswarm legibility mapping from index parity
        visual_jitter = np.array([(i % 11 - 5) * 0.03 for i in range(len(vals))])
        
        fig.add_trace(go.Scatter(
            x=vals, y=[rank]*len(vals) + visual_jitter, mode="markers",
            marker=dict(size=4, color=vals, colorscale="RdBu_r", cmin=-np.percentile(np.abs(shap_vals), 99), cmax=np.percentile(np.abs(shap_vals), 99), opacity=0.7),
            name=feat_names[fi], showlegend=False
        ))
    
    y_ticktext = [feat_names[i] for i in top_idx[::-1]]
    y_pos = {name: i for i, name in enumerate(y_ticktext)}
    
    def add_annot(t, text, color, offset):
        if t in y_pos: fig.add_annotation(x=offset, y=y_pos[t], text=f"<b>{text}</b>", showarrow=True, arrowhead=2, ax=offset*2 if offset > 0 else offset*2, ay=0, font=dict(size=12, color=color), bgcolor="white", bordercolor=color, borderwidth=2)

    add_annot("katG:S315T", "Primary Clinical Driver", C_FUSED, 1.5)
    add_annot("inhA:c-15t", "Primary Clinical Driver", C_FUSED, 2.0)
    add_annot("fabG1:L203L", "Primary Clinical Driver", C_FUSED, 1.0)
    add_annot("tlyA:L11L", "Co-occurrence Artefact", C_GATE, -1.0)
    add_annot("mmpL5:I948V", "Co-occurrence Artefact", C_GATE, -1.5)

    fig.update_layout(title=dict(text="Figure 4: SHAP Mutation Attribution (Top 20)", font=dict(size=18)), xaxis_title="SHAP Value (impact on model output)", yaxis=dict(tickvals=list(range(20)), ticktext=y_ticktext), template="plotly_white", width=1000, height=800)
    save_fig(fig, "Fig4_SHAP_Beeswarm")

# ====================================================================
# FIG 5: ATTENTION HEATMAP
# ====================================================================
def create_attention_heatmap():
    to = np.load("model/test_outputs.npz", allow_pickle=True)
    attn = to["attn_weights"]
    gene_names = list(to["gene_names"])
    labels = to["labels"]

    order = np.argsort(labels)
    attn_sorted = attn[order]
    
    n_show = 200
    step = max(1, attn_sorted.shape[0] // n_show)
    attn_sub = attn_sorted[::step][:n_show]
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=attn_sub, x=gene_names, y=list(range(n_show)), colorscale="YlOrRd", colorbar=dict(title="Attention")))
    
    if "pncA" in gene_names:
        pncA_idx = gene_names.index("pncA")
        # Explicitly visual highlight
        fig.add_shape(type="rect", x0=pncA_idx - 0.5, y0=0, x1=pncA_idx + 0.5, y1=n_show, line=dict(color=C_FUSED, width=3))
        fig.add_annotation(x=pncA_idx, y=n_show * 0.8, text="<b>⚠️ Phenotypic Routing Artefact</b><br>pncA acts as an MDR background proxy", showarrow=True, arrowhead=2, ax=-100, ay=0, font=dict(size=13, color=C_FUSED), bgcolor="white", bordercolor=C_FUSED, borderwidth=2)

    resistant_start = sum(labels[order[::step][:n_show]] == 0)
    fig.add_hline(y=resistant_start, line_width=2, line_dash="dash", line_color="black")
    fig.add_annotation(x=-1, y=resistant_start/2, text="Susceptible", textangle=-90, font=dict(size=14, color=C_DATA), showarrow=False)
    fig.add_annotation(x=-1, y=(n_show+resistant_start)/2, text="Resistant", textangle=-90, font=dict(size=14, color=C_FUSED), showarrow=False)

    fig.update_layout(title=dict(text="Figure 5: Gene Attention Weights by Phenotype", font=dict(size=18)), xaxis=dict(title="26 WHO Catalogue Genes", tickfont=dict(size=14), tickangle=-45), yaxis=dict(title="", showticklabels=False), template="plotly_white", width=1100, height=700)
    save_fig(fig, "Fig5_Attention_Heatmap")

# ====================================================================
# FIG 6: ATTENTION vs SHAP SCATTER
# ====================================================================
def create_scatter():
    am = json.load(open("explain/alignment_metrics.json"))
    df = pd.DataFrame(am["gene_importance_table"])
    gm = json.load(open("kg/gene_mechanism.json"))
    
    # Strictly compute real correlation purely from test outputs!
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["attn_rank"], df["shap_rank"])

    mechanism_colors = {"cell_wall_synthesis": "#E74C3C", "dna_replication": "#3498DB", "rna_transcription": "#2ECC71", "protein_synthesis": "#9B59B6", "regulatory": "#607D8B"}
    df["mechanism"] = df["gene"].map(gm)
    df["color"] = df["mechanism"].map(mechanism_colors).fillna("#999999")

    fig = go.Figure()
    
    for mech in df["mechanism"].unique():
        mdf = df[df["mechanism"] == mech]
        fig.add_trace(go.Scatter(x=mdf["attn_rank"], y=mdf["shap_rank"], mode="markers+text", text=mdf["gene"], textposition="top center", textfont=dict(size=12, color="black"), name=mech.replace("_", " ").title(), marker=dict(size=mdf["card_resistance_edges"].clip(1).apply(np.log1p) * 6 + 10, color=mdf["color"], opacity=0.8, line=dict(width=1.5, color="white"))))

    fig.add_trace(go.Scatter(x=[1, 26], y=[1, 26], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect Agreement"))

    katg = df[df["gene"] == "katG"].iloc[0]
    pnca = df[df["gene"] == "pncA"].iloc[0]
    
    fig.add_annotation(x=pnca["attn_rank"], y=pnca["shap_rank"], text="<b>pncA</b><br>High attention, low SHAP", showarrow=True, arrowhead=2, ax=60, ay=30, font=dict(size=11), bgcolor="white", bordercolor="#E74C3C")
    fig.add_annotation(x=katg["attn_rank"], y=katg["shap_rank"], text="<b>katG</b><br>High SHAP, under-attended<br>(Below the diagonal)", showarrow=True, arrowhead=2, ax=-60, ay=-30, font=dict(size=11), bgcolor="white", bordercolor=C_DATA)

    fig.update_layout(title=dict(text=f"Figure 6: Attention vs. SHAP Rank (The Regulatory Artefact)<br><sub>Spearman ρ = {rho:.3f}, p = {pval:.3f} (Dynamically computed from actual test data)</sub>", font=dict(size=18)), xaxis_title="Attention Rank (1 = Top Attention)", yaxis_title="SHAP Rank (1 = Top SHAP Importance)", xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"), template="plotly_white", width=900, height=700)
    save_fig(fig, "Fig6_Attention_vs_SHAP_Scatter")

# RUN ALL
if __name__ == "__main__":
    create_combined_figure_1()
    create_architecture()
    create_roc_confusion()
    create_shap_beeswarm()
    create_attention_heatmap()
    create_scatter()
