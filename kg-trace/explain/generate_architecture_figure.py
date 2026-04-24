"""
Recreates the KG-TRACE architecture diagram with two corrections:
  1. Gate α is explicitly shown as 128-dim element-wise vector (⊙ operator)
  2. Gate MLP internals + concatenation input are labeled
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_publication_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
BG          = "#F5EDD6"
KG_OUTER    = "#4A7BA7"
KG_INNER    = "#D6E8F5"
GEN_OUTER   = "#4A8C5C"
GEN_INNER   = "#D6F0E0"
FUSE_BG     = "#E8A020"
FUSE_BOX    = "#F5C060"
FUSE_DARK   = "#CC8010"
OUT_BG      = "#C05070"
OUT_BOX     = "#E8A0B0"
OUT_DARK    = "#903050"
ARROW_COL   = "#333333"
WHITE       = "#FFFFFF"
CREAM       = "#FFF8EC"

fig, ax = plt.subplots(figsize=(16, 11))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")

def box(ax, x, y, w, h, facecolor, edgecolor, radius=0.25, lw=2, zorder=2, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=lw, zorder=zorder, alpha=alpha)
    ax.add_patch(rect)
    return rect

def txt(ax, x, y, s, size=10, color="black", bold=False, center=True, zorder=5, wrap=False):
    weight = "bold" if bold else "normal"
    ha = "center" if center else "left"
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", zorder=zorder,
            multialignment="center" if center else "left")

def arrow(ax, x1, y1, x2, y2, color=ARROW_COL, lw=1.8, style="->", zorder=4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

def label_on_arrow(ax, x, y, s, size=8, color="#555555", bg=CREAM):
    ax.text(x, y, s, fontsize=size, color=color, ha="center", va="center",
            zorder=6, bbox=dict(boxstyle="round,pad=0.2", facecolor=bg,
                                edgecolor="#AAAAAA", linewidth=0.8))

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8, 10.5, "KG-TRACE: Knowledge Graph for Trustworthy\nAMR Classification & Explanation",
        fontsize=16, fontweight="bold", ha="center", va="center", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION HEADERS
# ══════════════════════════════════════════════════════════════════════════════
txt(ax, 2.9, 9.85, "Inputs / Encoders", size=11, bold=True, color="#333333")
txt(ax, 7.5, 9.85, "Fusion", size=11, bold=True, color="#333333")
txt(ax, 13.3, 9.85, "Outputs", size=11, bold=True, color="#333333")

# Divider lines
ax.plot([5.15, 5.15], [0.3, 9.7], color="#BBBBBB", lw=1, ls="--", zorder=1)
ax.plot([10.2, 10.2], [0.3, 9.7], color="#BBBBBB", lw=1, ls="--", zorder=1)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT ICONS (left strip)
# ══════════════════════════════════════════════════════════════════════════════
# WHO icon
box(ax, 0.15, 7.5, 0.85, 0.7, "#DDEEFF", "#4A7BA7", radius=0.1, lw=1.2)
txt(ax, 0.575, 7.98, "WHO", size=7, bold=True, color="#1A3A6A")
txt(ax, 0.575, 7.75, "Mutation", size=6.5, color="#333333")
txt(ax, 0.575, 7.55, "Catalogue", size=6.5, color="#333333")

# WGS icon
box(ax, 0.15, 4.3, 0.85, 0.7, "#DDEEFF", "#4A8C5C", radius=0.1, lw=1.2)
txt(ax, 0.575, 4.78, "WGS", size=7, bold=True, color="#1A4A1A")
txt(ax, 0.575, 4.55, "Isolate", size=6.5, color="#333333")
txt(ax, 0.575, 4.35, "Data", size=6.5, color="#333333")

# AST/DST icon
box(ax, 0.15, 1.5, 0.85, 0.7, "#DDEEFF", "#888888", radius=0.1, lw=1.2)
txt(ax, 0.575, 1.98, "AST/", size=7, bold=True, color="#333333")
txt(ax, 0.575, 1.75, "DST", size=7, bold=True, color="#333333")
txt(ax, 0.575, 1.55, "Labels", size=6.5, color="#333333")

# ══════════════════════════════════════════════════════════════════════════════
# KG BRANCH
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 1.1, 6.8, 4.0, 2.7, KG_OUTER, KG_OUTER, radius=0.3, lw=2.5, zorder=2)
txt(ax, 3.1, 9.25, "KG Branch", size=11, bold=True, color=WHITE, zorder=5)

box(ax, 1.4, 7.1, 3.4, 2.1, KG_INNER, KG_OUTER, radius=0.2, lw=1.5, zorder=3)
txt(ax, 3.1, 8.9, "RotatE Embeddings", size=9.5, bold=True, color="#1A4A6A", zorder=5)
# simple network icon
for ex, ey in [(2.5,8.5),(3.0,8.6),(3.5,8.5),(2.75,8.2),(3.25,8.2)]:
    ax.plot(ex, ey, "o", ms=7, color=KG_OUTER, zorder=5)
for (x1,y1),(x2,y2) in [((2.5,8.5),(3.0,8.6)),((3.0,8.6),(3.5,8.5)),
                          ((2.5,8.5),(2.75,8.2)),((3.0,8.6),(2.75,8.2)),
                          ((3.0,8.6),(3.25,8.2)),((3.5,8.5),(3.25,8.2))]:
    ax.plot([x1,x2],[y1,y2], color=KG_OUTER, lw=1.5, zorder=4)

txt(ax, 3.1, 7.55, "Gene Embeddings", size=8.5, bold=False, color="#1A4A6A", zorder=5)
txt(ax, 3.1, 7.28, "[B, 26, 64]", size=8, color="#2A5A7A", zorder=5)

# Arrow: WHO → KG branch
arrow(ax, 1.0, 7.85, 1.1, 7.85, color=KG_OUTER, lw=1.5)

# ══════════════════════════════════════════════════════════════════════════════
# GENOMIC BRANCH
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 1.1, 1.2, 4.0, 5.3, GEN_OUTER, GEN_OUTER, radius=0.3, lw=2.5, zorder=2)
txt(ax, 3.1, 6.25, "Genomic Branch", size=11, bold=True, color=WHITE, zorder=5)

# Binary Mutation Matrix sub-box
box(ax, 1.4, 4.2, 1.5, 2.1, GEN_INNER, GEN_OUTER, radius=0.2, lw=1.5, zorder=3)
txt(ax, 2.15, 5.55, "Binary", size=8.5, bold=True, color="#1A4A2A", zorder=5)
txt(ax, 2.15, 5.28, "Mutation", size=8.5, bold=True, color="#1A4A2A", zorder=5)
txt(ax, 2.15, 5.02, "Matrix", size=8.5, bold=True, color="#1A4A2A", zorder=5)
# mini matrix icon
for r in range(4):
    for c in range(4):
        val = (r+c) % 2
        ax.add_patch(mpatches.Rectangle((1.65+c*0.22, 4.35+r*0.22), 0.2, 0.2,
                     facecolor="#4A8C5C" if val else "#D6F0E0",
                     edgecolor="#4A8C5C", lw=0.5, zorder=4))

# Genomic Encoder sub-box
box(ax, 3.05, 4.2, 1.8, 2.1, GEN_INNER, GEN_OUTER, radius=0.2, lw=1.5, zorder=3)
txt(ax, 3.95, 5.9, "Genomic", size=8.5, bold=True, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 5.65, "Encoder", size=8.5, bold=True, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 5.35, "Linear(d,512)", size=7.5, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 5.12, "→BN→ReLU", size=7.5, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 4.89, "→Drop(0.3)", size=7.5, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 4.66, "→Linear(512,256)", size=7.5, color="#1A4A2A", zorder=5)
txt(ax, 3.95, 4.38, "d = 17,352", size=7, color="#2A6A3A", zorder=5)

# Arrow: Binary Matrix → Genomic Encoder
arrow(ax, 2.9, 5.25, 3.05, 5.25, color=GEN_OUTER, lw=1.5)

# AST/DST → Genomic bottom
arrow(ax, 1.0, 1.85, 3.95, 1.85, color="#888888", lw=1.5)
arrow(ax, 3.95, 1.85, 3.95, 1.5, color="#888888", lw=1.5)  # down to bottom of box

# WGS → Binary Mutation
arrow(ax, 1.0, 4.65, 1.4, 4.65, color=GEN_OUTER, lw=1.5)

# Output g [256-dim] label below genomic encoder
txt(ax, 3.95, 3.75, "g  [256-dim]", size=8.5, bold=True, color=WHITE, zorder=5,
    )
# small arrow g going right (out of genomic branch)
arrow(ax, 3.95, 3.55, 3.95, 1.5, color=GEN_OUTER, lw=1.2)  # dummy, actual is rightward

# ══════════════════════════════════════════════════════════════════════════════
# FUSION SECTION BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 5.25, 0.5, 4.9, 9.0, "#F5D070", FUSE_DARK, radius=0.4, lw=2.5, alpha=0.35, zorder=1)
txt(ax, 7.7, 9.3, "Cross-Attention Fusion", size=11, bold=True, color="#7A5000", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# CROSS-ATTENTION POOLING BOX
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 5.55, 6.6, 4.3, 2.3, FUSE_BOX, FUSE_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, 7.7, 8.5, "Cross-Attention Pooling", size=10, bold=True, color="#5A3000", zorder=5)
txt(ax, 7.7, 8.15, "query = gene_attn_q(g)  [B,64,1]", size=7.8, color="#5A3000", zorder=5)
txt(ax, 7.7, 7.88, "scores = bmm(gene_embeds, query) / √64", size=7.8, color="#5A3000", zorder=5)
txt(ax, 7.7, 7.61, "α_attn = softmax(scores)  [B,26,1]", size=7.8, color="#5A3000", zorder=5)
txt(ax, 7.7, 7.34, "k = Σ α_attn · gene_embeds  [B,64]", size=7.8, color="#5A3000", zorder=5)
txt(ax, 7.7, 6.95, "proj_k : 64 → 128", size=8, color="#8B4000", bold=True, zorder=5)

# k [64-dim] label on arrow from KG to cross-attn
label_on_arrow(ax, 5.35, 7.75, "k [64-dim]", size=8)

# ══════════════════════════════════════════════════════════════════════════════
# SIGMOID FUSION GATE BOX  (CORRECTED)
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 5.55, 2.0, 4.3, 4.3, FUSE_BOX, FUSE_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, 7.7, 5.95, "Sigmoid Fusion Gate  (MLP + σ)", size=10, bold=True, color="#5A3000", zorder=5)

# ── CORRECTION 2: concatenation input + gate MLP details ──────────────────
box(ax, 5.85, 5.35, 3.7, 0.52, "#FDEBD0", "#CC8010", radius=0.12, lw=1.5, zorder=4)
txt(ax, 7.7, 5.63, "[g_proj ⊕ k_proj]  →  256-dim  (concat)", size=8, bold=True,
    color="#7A4000", zorder=5)

txt(ax, 7.7, 5.05, "Gate MLP internals:", size=8, bold=True, color="#5A3000", zorder=5)
txt(ax, 7.7, 4.78, "Linear(256 → 128)  →  ReLU  →  Linear(128 → 128)  →  Sigmoid",
    size=7.8, color="#5A3000", zorder=5)

# proj_g: 256→128 label
txt(ax, 7.7, 4.42, "proj_g : 256 → 128        proj_k : 64 → 128", size=8,
    color="#8B4000", bold=True, zorder=5)
txt(ax, 7.7, 4.15, "g_proj [128-dim]             k_proj [128-dim]", size=7.8,
    color="#5A3000", zorder=5)

# Separator line
ax.plot([5.75, 9.65], [3.92, 3.92], color=FUSE_DARK, lw=1, ls="--", zorder=4, alpha=0.5)

# ── CORRECTION 1: formula with ⊙ element-wise operator + dimension note ──
box(ax, 5.85, 2.15, 3.7, 1.65, "#FFF0D0", "#CC8010", radius=0.12, lw=1.5, zorder=4)
txt(ax, 7.7, 3.65, "Fused Representation  h  [128-dim]", size=9, bold=True,
    color="#7A4000", zorder=5)
txt(ax, 7.7, 3.32,
    r"$\mathbf{h} = \boldsymbol{\alpha} \odot \mathbf{g}_{\mathrm{proj}}"
    r" + (\mathbf{1}-\boldsymbol{\alpha}) \odot \mathbf{k}_{\mathrm{proj}}$",
    size=10, color="#5A3000", zorder=5)
# correction annotation
box(ax, 5.85, 2.18, 3.7, 0.75, "#FFF0D0", "#CC8010", radius=0.1, lw=0, zorder=3)
txt(ax, 7.7, 2.75,
    "α ∈ (0,1)¹²⁸  — element-wise gate vector  (NOT scalar)",
    size=8, color="#AA3300", bold=True, zorder=5)
txt(ax, 7.7, 2.42,
    "Each of the 128 dims gets its own independent gate value",
    size=7.5, color="#663300", zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS inside fusion section
# ══════════════════════════════════════════════════════════════════════════════
# Cross-Attn Pooling → Gate (k_proj)
arrow(ax, 7.7, 6.6, 7.7, 6.35, color=FUSE_DARK, lw=2)
label_on_arrow(ax, 7.7, 6.47, "k_proj [128-dim]", size=8)

# g_proj → Gate
arrow(ax, 7.0, 6.6, 7.0, 6.35, color=FUSE_DARK, lw=1.5, style="-")
# (g enters from bottom of cross-attn via side path)

# Fused output → out of gate box
arrow(ax, 7.7, 2.0, 7.7, 1.6, color=FUSE_DARK, lw=2)
label_on_arrow(ax, 7.7, 1.8, "h [128-dim]", size=8)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS: Encoder → Fusion
# ══════════════════════════════════════════════════════════════════════════════
# KG → Cross-Attention Pooling (gene embeddings enter from left)
arrow(ax, 5.1, 7.75, 5.55, 7.75, color=KG_OUTER, lw=2)
label_on_arrow(ax, 5.3, 8.0, "Gene Emb\n[B,26,64]", size=7.5, bg=KG_INNER)

# Genomic g → query arrow (diagonal, into cross-attn)
arrow(ax, 5.1, 5.25, 6.3, 7.5, color=GEN_OUTER, lw=1.8)
label_on_arrow(ax, 5.55, 6.2, "query\n(gene_attn_q:\n256→64)", size=7, bg=GEN_INNER)

# Genomic g → proj_g → gate (straight right into gate box)
arrow(ax, 5.1, 4.0, 5.55, 4.0, color=GEN_OUTER, lw=1.8)
label_on_arrow(ax, 5.3, 4.25, "g [256]", size=7.5, bg=GEN_INNER)

# KG → proj_k side path (enters gate from left at different y)
arrow(ax, 5.1, 7.1, 5.55, 5.7, color=KG_OUTER, lw=1.5, style="->")

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUTS SECTION
# ══════════════════════════════════════════════════════════════════════════════
out_x = 10.3

# Background
box(ax, out_x - 0.05, 0.5, 5.5, 9.0, "#F0D0D8", OUT_DARK, radius=0.4, lw=2.5, alpha=0.3, zorder=1)

# Primary Loss label
ax.annotate("Primary Loss\n(CE)", xy=(10.3, 7.2), xytext=(9.4, 7.5),
            fontsize=8, color="#903050",
            arrowprops=dict(arrowstyle="->", color="#903050", lw=1.2))

# R/S Prediction box
box(ax, out_x, 7.8, 5.0, 1.5, OUT_BOX, OUT_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, out_x+2.5, 8.8, "R / S  Prediction", size=11, bold=True, color=WHITE, zorder=5)
txt(ax, out_x+2.5, 8.4, "AMR Classifier", size=9, bold=True, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 8.1, "Linear(128,64) → ReLU → Linear(64,2)", size=8, color="#5A0020", zorder=5)

# Level 1 SHAP
box(ax, out_x, 5.9, 5.0, 1.6, OUT_BOX, OUT_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, out_x+2.5, 7.15, "Level 1 — SHAP Attribution", size=9.5, bold=True, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 6.82, "(Primary Trust Layer)", size=8.5, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 6.48, "Post-hoc feature attribution via SHAP", size=7.5, color="#7A2040", zorder=5)
txt(ax, out_x+2.5, 6.18, "→ per-mutation importance ranking", size=7.5, color="#7A2040", zorder=5)

# Level 2 KG Pathway
box(ax, out_x, 4.0, 5.0, 1.6, OUT_BOX, OUT_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, out_x+2.5, 5.25, "Level 2 — KG Pathway Coverage", size=9.5, bold=True, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 4.92, "(Mechanistic Layer)", size=8.5, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 4.58, "Cross-attention weights → gene relevance", size=7.5, color="#7A2040", zorder=5)
txt(ax, out_x+2.5, 4.28, "→ pathway/mechanism attribution", size=7.5, color="#7A2040", zorder=5)

# Auxiliary Loss label
ax.annotate("Auxiliary Loss\n(BCE, λ=0.3)", xy=(10.3, 2.5), xytext=(9.4, 2.2),
            fontsize=8, color="#903050",
            arrowprops=dict(arrowstyle="->", color="#903050", lw=1.2))

# Gene Auxiliary Head
box(ax, out_x, 1.0, 5.0, 2.7, OUT_BOX, OUT_DARK, radius=0.25, lw=2, zorder=3)
txt(ax, out_x+2.5, 3.3, "Gene Auxiliary Head", size=10, bold=True, color=WHITE, zorder=5)
txt(ax, out_x+2.5, 2.95, "Detection of 26 WHO Genes", size=8.5, bold=True, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 2.65, "Linear(128,64) → ReLU", size=8, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 2.4,  "→ Linear(64,26) → Sigmoid", size=8, color="#5A0020", zorder=5)
txt(ax, out_x+2.5, 2.1,  "(multi-label BCE loss, λ=0.3)", size=7.5, color="#7A2040", zorder=5)
txt(ax, out_x+2.5, 1.75, "Auxiliary supervision to guide", size=7.5, color="#7A2040", zorder=5)
txt(ax, out_x+2.5, 1.45, "attention to known resistance genes", size=7.5, color="#7A2040", zorder=5)

# ── Arrows: fusion h → outputs ─────────────────────────────────────────────
# h → AMR classifier
arrow(ax, 10.2, 8.3, out_x, 8.3, color=OUT_DARK, lw=2)
# h → Level 1 SHAP
arrow(ax, 10.2, 7.1, out_x, 7.1, color=OUT_DARK, lw=1.8)
# h → Level 2 KG
arrow(ax, 10.2, 5.2, out_x, 5.2, color=OUT_DARK, lw=1.8)
# h → Gene Head
arrow(ax, 10.2, 2.35, out_x, 2.35, color=OUT_DARK, lw=2)

# Vertical line connecting all outputs from h
ax.plot([10.2, 10.2], [1.6, 8.55], color=OUT_DARK, lw=2, zorder=3)
ax.plot([9.85, 10.2], [1.6, 1.6], color=OUT_DARK, lw=1.5, zorder=3)

# ── Correction callout boxes ───────────────────────────────────────────────
# Callout 1: α is element-wise
box(ax, 5.7, 0.55, 4.6, 0.6, "#FFEECC", "#DD6600", radius=0.1, lw=1.5, zorder=6)
ax.text(5.9, 0.87, "★ Correction 1:", fontsize=7.5, color="#DD4400", fontweight="bold",
        va="center", zorder=7)
ax.text(5.9, 0.67, "α is a 128-dim element-wise gate vector  — ⊙ denotes element-wise multiply",
        fontsize=7, color="#AA3300", va="center", zorder=7)

# Callout 2: gate MLP concatenation
box(ax, 5.7, 0.0, 4.6, 0.5, "#FFEECC", "#DD6600", radius=0.1, lw=1.5, zorder=6)
ax.text(5.9, 0.27, "★ Correction 2:", fontsize=7.5, color="#DD4400", fontweight="bold",
        va="center", zorder=7)
ax.text(5.9, 0.1, "Gate input = concat([g_proj, k_proj]) → 256-dim; MLP: Lin(256→128)→ReLU→Lin(128→128)→σ",
        fontsize=7, color="#AA3300", va="center", zorder=7)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
plt.tight_layout(pad=0.5)
out_png = f"{OUT_DIR}/Fig1_Architecture_corrected.png"
out_pdf = f"{OUT_DIR}/Fig1_Architecture_corrected.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=BG)
fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
