"""
ROC + Ablation figure
Left panel : Full ROC curves – KG-TRACE, SVM, XGBoost, RandomForest
Right panel: AUROC bar chart – 5 ablation configurations (100-epoch runs)
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc

# ── Paths ────────────────────────────────────────────────────────────────────
BASE  = "/Users/namangarg/Desktop/AMR NamanXSarika/kg-amr-v2"
MODEL_NPZ    = f"{BASE}/model/test_outputs.npz"
BASELINE_NPZ = f"{BASE}/baselines/baseline_predictions.npz"
REVIEWER_JSON = f"{BASE}/evaluate/reviewer_experiments/all_reviewer_experiments.json"
OUT_PNG = f"{BASE}/explain/final_publication_figures/Fig_ROC.png"
OUT_PDF = f"{BASE}/explain/final_publication_figures/Fig_ROC.pdf"

# ── Load primary model ────────────────────────────────────────────────────────
td     = np.load(MODEL_NPZ, allow_pickle=True)
y_true = td["labels"]
y_prob = td["probs"]

# ── Load baselines ────────────────────────────────────────────────────────────
bd          = np.load(BASELINE_NPZ, allow_pickle=True)
svm_probs   = bd["svm_probs"]
xgb_probs   = bd["xgb_probs"]
rf_probs    = bd["rf_probs"]
y_test_base = bd["y_test"]

# ── Styling ───────────────────────────────────────────────────────────────────
COLORS = {
    "KG-TRACE":           "#1a6faf",
    "SVM (LinearSVC)":    "#e07b39",
    "XGBoost":            "#2e9e4f",
    "Random Forest":      "#c0392b",
}
# ── Figure layout ─────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 5.5))
fig.patch.set_facecolor("white")

# ═════════════════════════════════════════════════════════════════════════════
# ROC Curves
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc(ax, y_true, y_score, label, color, lw=2, ls="-", zorder=3):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, zorder=zorder,
            label=f"{label}  (AUROC = {roc_auc:.4f})")
    return roc_auc

# Diagonal
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, zorder=1)

# Primary model (thick, on top)
plot_roc(ax1, y_true,     y_prob,   "KG-TRACE",        COLORS["KG-TRACE"],        lw=2.5, zorder=5)
plot_roc(ax1, y_test_base, svm_probs, "SVM (LinearSVC)", COLORS["SVM (LinearSVC)"], lw=1.8, ls="--")
plot_roc(ax1, y_test_base, xgb_probs, "XGBoost",         COLORS["XGBoost"],         lw=1.8, ls="-.")
plot_roc(ax1, y_test_base, rf_probs,  "Random Forest",   COLORS["Random Forest"],   lw=1.8, ls=":")

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.02])
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate", fontsize=12)
ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.92, edgecolor="#cccccc")
ax1.grid(True, linestyle="--", alpha=0.35)

# Inset zoom: top-left corner (high-sensitivity region)
axins = ax1.inset_axes([0.08, 0.46, 0.42, 0.40])
plot_roc(axins, y_true,     y_prob,    "KG-TRACE",        COLORS["KG-TRACE"],        lw=2.0, zorder=5)
plot_roc(axins, y_test_base, svm_probs, "SVM",             COLORS["SVM (LinearSVC)"], lw=1.5, ls="--")
plot_roc(axins, y_test_base, xgb_probs, "XGBoost",         COLORS["XGBoost"],         lw=1.5, ls="-.")
plot_roc(axins, y_test_base, rf_probs,  "RF",              COLORS["Random Forest"],   lw=1.5, ls=":")
axins.set_xlim(0.0, 0.12)
axins.set_ylim(0.88, 1.01)
axins.set_xticks([0.0, 0.05, 0.10])
axins.set_yticks([0.90, 0.95, 1.00])
axins.tick_params(labelsize=7.5)
axins.set_title("zoom", fontsize=8, pad=2)
axins.grid(True, linestyle="--", alpha=0.3)
axins.get_legend().remove() if axins.get_legend() else None
ax1.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.5)

# ═════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Ablation AUROC Bar Chart
# ═════════════════════════════════════════════════════════════════════════════

# Order: full model first, then descending AUROC
ablation_100ep_sorted = sorted(ablation_100ep, key=lambda x: x["auroc"], reverse=True)

configs   = [d["config"] for d in ablation_100ep_sorted]
aurocs    = [d["auroc"]  for d in ablation_100ep_sorted]
f1s       = [d["f1_macro"] for d in ablation_100ep_sorted]
epochs    = [d["stopped_epoch"] for d in ablation_100ep_sorted]
bar_colors = [ABLATION_COLORS[c] for c in configs]
labels    = [ABLATION_LABELS[c]  for c in configs]

x = np.arange(len(configs))
bar_w = 0.38

bars_auroc = ax2.bar(x - bar_w/2, aurocs, bar_w, color=bar_colors,
                     edgecolor="white", linewidth=0.8, label="AUROC", zorder=3)
bars_f1    = ax2.bar(x + bar_w/2, f1s,    bar_w, color=bar_colors,
                     edgecolor="white", linewidth=0.8, alpha=0.55, label="F1-macro",
                     hatch="///", zorder=3)

# Value labels on bars
for bar in bars_auroc:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}",
             ha="center", va="bottom", fontsize=7.5, fontweight="bold")
for bar in bars_f1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.4f}",
             ha="center", va="bottom", fontsize=7.5, color="#444444")

# Epoch annotations below bars
for i, (cfg, ep) in enumerate(zip(configs, epochs)):
    ax2.text(x[i], 0.895, f"ep {ep}", ha="center", va="bottom",
             fontsize=7, color="#666666", style="italic")

ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9.5)
ax2.set_ylim(0.90, 1.005)
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("(B)  Ablation Study — AUROC & F1-macro\n"
              "(100-epoch runs, early stopping; epoch at convergence shown)",
              fontsize=12, pad=8)
ax2.grid(True, axis="y", linestyle="--", alpha=0.35, zorder=0)
ax2.yaxis.set_tick_params(labelsize=10)

# Legend for bar shading
solid_patch  = mpatches.Patch(facecolor="#888888", edgecolor="white", label="AUROC (solid)")
hatch_patch  = mpatches.Patch(facecolor="#888888", edgecolor="white", alpha=0.55,
                               hatch="///", label="F1-macro (hatched)")
ax2.legend(handles=[solid_patch, hatch_patch], fontsize=9, loc="lower right",
           framealpha=0.92, edgecolor="#cccccc")

# Highlight best config
best_idx = aurocs.index(max(aurocs))
ax2.get_xticklabels()[best_idx].set_color("#1a6faf")
ax2.get_xticklabels()[best_idx].set_fontweight("bold")

# ── Final tweaks ──────────────────────────────────────────────────────────────
plt.tight_layout(pad=2.5)

import os
os.makedirs(f"{BASE}/explain/final_publication_figures", exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_PDF, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")

# Print summary table
print("\n=== ABLATION SUMMARY (sorted by AUROC) ===")
print(f"{'Config':<26} {'AUROC':>8} {'F1':>8} {'Stopped Ep':>11}")
for d in ablation_100ep_sorted:
    print(f"{ABLATION_LABELS[d['config']]:<26} {d['auroc']:>8.4f} {d['f1_macro']:>8.4f} {d['stopped_epoch']:>11}")
