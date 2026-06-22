"""
Regenerate SHAP values from the INH v1 checkpoint, then compute BGR@k and gate stats.
All numbers logged to JSON for paper audit trail.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from paths import KG_EMBED_DIM, FUSED_DIM, PROJECT_DIR

import numpy as np
import torch
import networkx as nx
from scipy import sparse, stats

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")

DRUG = "INH"
CKPT = os.path.join(MODEL_DIR, "checkpoints/ablation_genomic_only/best.ckpt")

# ── Load test_outputs (just regenerated from INH v1) ──
print("[1/5] Loading INH test outputs...")
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
gene_names = test_data["gene_names"]
gene_presence = test_data["gene_presence"]
labels = test_data["labels"]
preds = test_data["preds"]
probs = test_data["probs"]
gate_values = test_data["gate_values"]
n_test = len(test_ids)
print(f"  {n_test} samples, {(labels==1).sum()} R / {(labels==0).sum()} S")

# ── Load model for SHAP ──
print("[2/5] Loading INH model and computing gradient×input SHAP...")
from model.kg_amr import KGTrace

with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)

KMER_DIM = len(all_features)
NUM_GENES = len(gene_names)

model = KGTrace.load_from_checkpoint(CKPT, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
model.eval()
model.to('cpu')

# Load X_test
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_sample_indices = [sample_to_idx[s] for s in test_ids]
X_test = X_sparse[test_sample_indices].toarray().astype(np.float32)

# Load gene embeddings
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
entity_emb = np.abs(entity_emb_raw).astype(np.float32) if np.iscomplexobj(entity_emb_raw) else entity_emb_raw.astype(np.float32)
with open(os.path.join(KG_DIR, "embeddings/pykeen_entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(KG_DIR, "gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
CATALOGUE_GENES = sorted(gene_mechanism.keys())
gene_emb_indices = [entity_to_id[g] for g in CATALOGUE_GENES]
gene_emb_matrix = entity_emb[gene_emb_indices]

gene_embeds_test = np.zeros((n_test, NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(n_test):
    for j in range(NUM_GENES):
        if gene_presence[i, j] > 0:
            gene_embeds_test[i, j, :] = gene_emb_matrix[j]

# Compute SHAP
t0 = time.time()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
ge_test_tensor = torch.tensor(gene_embeds_test, dtype=torch.float32)
shap_values = np.zeros_like(X_test)

with torch.enable_grad():
    for i in range(n_test):
        if i % 1000 == 0:
            print(f"  [{i}/{n_test}]...")
        x_i = X_test_tensor[i:i+1].clone().detach().requires_grad_(True)
        ge_i = ge_test_tensor[i:i+1]
        logits, _, _, _ = model(x_i, ge_i)
        logits[0, 1].backward()
        grad = x_i.grad.detach().numpy()[0]
        shap_values[i] = grad * X_test[i]

shap_time = time.time() - t0
print(f"  SHAP computed in {shap_time:.1f}s ({shap_time/n_test*1000:.1f}ms/sample)")

# Save
np.savez(os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"),
         shap_values=shap_values, feature_names=np.array(all_features))
print(f"  Saved shap_raw_values.npz")

# ── BGR at multiple k ──
print("\n[3/5] Computing BGR@k...")
G = nx.read_graphml(os.path.join(PROJECT_DIR, "kg/amr_graph.graphml"))

mean_abs_shap = np.abs(shap_values).mean(axis=0)
sorted_indices = np.argsort(mean_abs_shap)[::-1]

cache = {}
def has_kg_path(feat_name):
    if feat_name in cache:
        return cache[feat_name]
    result = 0
    if G.has_node(feat_name) and G.has_node(DRUG) and nx.has_path(G, feat_name, DRUG):
        result = 1
    else:
        gene = feat_name.split(":")[0]
        if G.has_node(gene) and G.has_node(DRUG) and nx.has_path(G, gene, DRUG):
            result = 1
    cache[feat_name] = result
    return result

bgr_results = {}
for k in [10, 20, 50, 100]:
    top_k = [all_features[i] for i in sorted_indices[:k]]
    grounded = sum(has_kg_path(f) for f in top_k)
    bgr = grounded / k
    bgr_results[f"BGR@{k}"] = {"grounded": grounded, "total": k, "ratio": round(bgr, 4)}
    print(f"  BGR@{k}: {grounded}/{k} = {bgr:.4f}")

# Top-10 features detail
print("\n  Top-10 global SHAP features:")
for rank, idx in enumerate(sorted_indices[:10]):
    f = all_features[idx]
    g = has_kg_path(f)
    print(f"    {rank+1}. {f}: |SHAP|={mean_abs_shap[idx]:.4f} {'GROUNDED' if g else 'NOT GROUNDED'}")

# ── Per-sample symbolic coverage ──
print("\n[4/5] Per-sample symbolic coverage (top-1 for predicted-R)...")
n_pred_r = (preds == 1).sum()
grounded_top1 = 0
for i in range(n_test):
    if preds[i] != 1:
        continue
    genome_shap = np.abs(shap_values[i])
    top_idx = np.argmax(genome_shap)
    top_feat = all_features[top_idx]
    if has_kg_path(top_feat):
        grounded_top1 += 1

coverage = grounded_top1 / n_pred_r if n_pred_r > 0 else 0
print(f"  {grounded_top1}/{n_pred_r} = {coverage*100:.1f}% symbolic coverage")

# ── Gate analysis ──
print("\n[5/5] Gate analysis...")
gate_means = gate_values.mean(axis=1).ravel()
confidences = np.maximum(probs.ravel(), 1 - probs.ravel())

r_gates = gate_means[labels == 1]
s_gates = gate_means[labels == 0]

# Per-sample strict BGR for correlation
strict_bgrs = []
flags = []
for i in range(n_test):
    genome_shap = np.abs(shap_values[i])
    genome_top = np.argsort(genome_shap)[::-1][:50]
    top_feats = [all_features[fi] for fi in genome_top]
    mapped = sum(has_kg_path(f) for f in top_feats)
    strict_bgr = mapped / 50.0
    strict_bgrs.append(strict_bgr)
    flags.append("HIGH" if strict_bgr > 0 else "UNCERTAIN")

strict_bgrs = np.array(strict_bgrs)
flags = np.array(flags)

rho_bgr, p_bgr = stats.spearmanr(gate_means, strict_bgrs)
rho_conf, p_conf = stats.spearmanr(gate_means, confidences)

gate_unc = gate_means[flags == "UNCERTAIN"].mean() if (flags == "UNCERTAIN").sum() > 0 else 0
gate_high = gate_means[flags == "HIGH"].mean() if (flags == "HIGH").sum() > 0 else 0

# ── Save all results ──
all_results = {
    "drug": DRUG,
    "checkpoint": CKPT,
    "n_test": int(n_test),
    "shap_time_seconds": round(shap_time, 1),
    "shap_ms_per_sample": round(shap_time / n_test * 1000, 1),
    "bgr": bgr_results,
    "symbolic_coverage_pct": round(coverage * 100, 1),
    "symbolic_coverage_detail": f"{grounded_top1}/{n_pred_r}",
    "gate": {
        "resistant_mean": round(float(r_gates.mean()), 4),
        "resistant_std": round(float(r_gates.std()), 4),
        "susceptible_mean": round(float(s_gates.mean()), 4),
        "susceptible_std": round(float(s_gates.std()), 4),
        "overall_mean": round(float(gate_means.mean()), 4),
        "overall_std": round(float(gate_means.std()), 4),
    },
    "gate_correlations": {
        "gate_vs_strict_bgr": {"rho": round(float(rho_bgr), 4), "p": float(p_bgr)},
        "gate_vs_confidence": {"rho": round(float(rho_conf), 4), "p": float(p_conf)},
    },
    "actionability": {
        "n_high": int((flags == "HIGH").sum()),
        "n_uncertain": int((flags == "UNCERTAIN").sum()),
        "gate_mean_high": round(float(gate_high), 4),
        "gate_mean_uncertain": round(float(gate_unc), 4),
    },
}

out_path = os.path.join(EXPLAIN_DIR, "inh_full_audit.json")
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nAll results saved to {out_path}")
print(json.dumps(all_results, indent=2))
print("DONE")
