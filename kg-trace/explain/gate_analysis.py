import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import spearmanr

# Directories
PROJECT_DIR = "/Users/namangarg/Desktop/KG-Trace/kg-trace"
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
FIG_DIR = os.path.join(PROJECT_DIR, "paper/figures")
os.makedirs(FIG_DIR, exist_ok=True)

DRUG = "INH"

print("[1/4] Loading outputs and SHAP values...")
# Load outputs
outputs = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
probs = outputs["probs"]
preds = outputs["preds"]
labels = outputs["labels"]
gate_values = outputs["gate_values"]

# Gate values: shape (N, 1), prob shape (N,)
gate_means = gate_values.mean(axis=1) if gate_values.ndim > 1 else gate_values
# Ensure it's 1D
gate_means = gate_means.ravel()
probs = probs.ravel()
labels = labels.ravel()
confidences = np.maximum(probs, 1 - probs)

# Load SHAP
shap_data = np.load(os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"), allow_pickle=True)
shap_values = shap_data["shap_values"]
feature_names = list(shap_data["feature_names"])

# Load KG
print("[2/4] Loading Knowledge Graph for Strict BGR...")
G = nx.read_graphml(os.path.join(PROJECT_DIR, "kg/amr_graph.graphml"))

print("[3/4] Computing Strict BGR per genome (Actionability Flags)...")
strict_bgrs = []
flags = []

n_test = len(labels)
cache = {}
for i in range(n_test):
    genome_shap = np.abs(shap_values[i])
    genome_top = np.argsort(genome_shap)[::-1][:50]
    top_feats = [feature_names[fi] for fi in genome_top]
    
    mapped = 0
    for f in top_feats:
        if f in cache:
            mapped += cache[f]
            continue
            
        is_mapped = 0
        if G.has_node(f) and G.has_node(DRUG) and nx.has_path(G, f, DRUG):
            is_mapped = 1
        else:
            gene = f.split(":")[0]
            if G.has_node(gene) and G.has_node(DRUG) and nx.has_path(G, gene, DRUG):
                is_mapped = 1
                
        cache[f] = is_mapped
        mapped += is_mapped
                
    strict_bgr = mapped / 50.0
    strict_bgrs.append(strict_bgr)
    flags.append("HIGH" if strict_bgr > 0 else "UNCERTAIN")

strict_bgrs = np.array(strict_bgrs)
flags = np.array(flags)

print(f"  UNCERTAIN flags: {np.sum(flags == 'UNCERTAIN')}")
print(f"  HIGH flags:      {np.sum(flags == 'HIGH')}")

print("\n[4/4] Generating Plots and Correlations...")

df = pd.DataFrame({
    "gate": gate_means,
    "prob": probs,
    "confidence": confidences,
    "label": ["Resistant" if l == 1 else "Susceptible" for l in labels],
    "strict_bgr": strict_bgrs,
    "actionability": flags
})

# Correlations
rho_conf, p_conf = spearmanr(df["gate"], df["confidence"])
rho_bgr, p_bgr = spearmanr(df["gate"], df["strict_bgr"])

print(f"  Gate vs Confidence: rho = {rho_conf:.4f}, p = {p_conf:.4e}")
print(f"  Gate vs Strict BGR: rho = {rho_bgr:.4f}, p = {p_bgr:.4e}")

# Mean values
gate_r = df[df["label"] == "Resistant"]["gate"].mean()
gate_s = df[df["label"] == "Susceptible"]["gate"].mean()
print(f"  Mean Gate (Resistant): {gate_r:.4f}")
print(f"  Mean Gate (Susceptible): {gate_s:.4f}")

# Handle case if UNCERTAIN is empty
gate_unc = df[df["actionability"] == "UNCERTAIN"]["gate"].mean() if np.sum(flags == 'UNCERTAIN') > 0 else 0.0
gate_high = df[df["actionability"] == "HIGH"]["gate"].mean() if np.sum(flags == 'HIGH') > 0 else 0.0
print(f"  Mean Gate (UNCERTAIN): {gate_unc:.4f}")
print(f"  Mean Gate (HIGH): {gate_high:.4f}")

# Save JSON
results = {
    "mean_gate_resistant": float(gate_r),
    "mean_gate_susceptible": float(gate_s),
    "mean_gate_uncertain": float(gate_unc),
    "mean_gate_high": float(gate_high),
    "corr_gate_confidence": {"rho": float(rho_conf), "pvalue": float(p_conf)},
    "corr_gate_strict_bgr": {"rho": float(rho_bgr), "pvalue": float(p_bgr)}
}
with open(os.path.join(EXPLAIN_DIR, "gate_analysis.json"), "w") as f:
    json.dump(results, f, indent=2)

# Plotting
sns.set_theme(style="whitegrid", context="paper")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Gate distribution by label
sns.boxplot(data=df, x="label", y="gate", ax=axes[0], palette="Set2")
axes[0].set_title("Fusion Gate ($\\alpha$) by True Label")
axes[0].set_ylabel("Gate $\\alpha$ (Genomic weight)")
axes[0].set_xlabel("")

# 2. Gate distribution by actionability
sns.boxplot(data=df, x="actionability", y="gate", ax=axes[1], palette="Set1", order=["HIGH", "UNCERTAIN"])
axes[1].set_title("Gate by Actionability Flag")
axes[1].set_ylabel("")
axes[1].set_xlabel("")

# 3. Gate vs Confidence
sns.scatterplot(data=df, x="confidence", y="gate", hue="label", alpha=0.3, ax=axes[2], palette="Set2")
axes[2].set_title(f"Gate vs Confidence\n(Spearman $\\rho={rho_conf:.2f}$)")
axes[2].set_xlabel("Prediction Confidence $\\max(p, 1-p)$")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "gate_analysis.pdf"), dpi=300, bbox_inches="tight")
print(f"Saved figure to {FIG_DIR}/gate_analysis.pdf")
print("DONE — gate_analysis.py")
