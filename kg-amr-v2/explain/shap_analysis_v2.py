"""
Step 8A-C (OPTIMIZED): Gene Attention, Fusion Gates, SHAP via Gradient×Input
Uses gradient·input approximation (10-50x faster than GradientExplainer on 17K features).
"""
import sys, os, time, json, csv, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, FUSED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "kg-amr-v2" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse

MODEL_DIR = os.path.join(PROJECT_DIR, "model")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
KG_DIR = os.path.join(PROJECT_DIR, "kg")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ── 1. Load model + test data ───────────────────────────────────────────────
print("[1/4] Loading model and test data...")
t0 = time.time()

# Load test outputs from training
test_data = np.load(os.path.join(MODEL_DIR, "test_outputs.npz"), allow_pickle=True)
test_ids = test_data["test_ids"]
gene_names = test_data["gene_names"]
attn_weights = test_data["attn_weights"]   # [n_test, 26]
gate_values = test_data["gate_values"]     # [n_test, FUSED_DIM]
preds = test_data["preds"]
probs = test_data["probs"]
labels = test_data["labels"]
gene_presence = test_data["gene_presence"] # [n_test, 26]

n_test = len(test_ids)
print(f"  Test samples: {n_test}")

# ── 2A. Gene Attention Weights ───────────────────────────────────────────────
print("\n[2/4] Extracting gene attention weights...")

attn_sums = attn_weights.sum(axis=1)
violations = np.where(np.abs(attn_sums - 1.0) > 1e-4)[0]
if len(violations) > 0:
    print(f"  WARNING: {len(violations)} genomes have attention sums != 1.0")
else:
    print(f"  All {n_test} attention weights sum to ~1.0 ✓")

attn_path = os.path.join(EXPLAIN_DIR, "gene_attention_weights.csv")
with open(attn_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genome_id", "gene_name", "attention_weight", "gene_present"])
    for i in range(n_test):
        for j, gname in enumerate(gene_names):
            writer.writerow([
                test_ids[i],
                gname,
                f"{attn_weights[i, j]:.6f}",
                int(gene_presence[i, j]),
            ])
print(f"  Saved {n_test * len(gene_names)} rows to gene_attention_weights.csv")

# ── 2B. Fusion Gate Values ───────────────────────────────────────────────────
print("\n[3/4] Extracting fusion gate values...")

gate_means = gate_values.mean(axis=1)
n_kg_dominated = int((gate_means < 0.5).sum())
n_genomic_dominated = int((gate_means >= 0.5).sum())
print(f"  {n_kg_dominated} ({100*n_kg_dominated/n_test:.1f}%) KG-dominated")
print(f"  {n_genomic_dominated} ({100*n_genomic_dominated/n_test:.1f}%) genomic-dominated")

gate_path = os.path.join(EXPLAIN_DIR, "fusion_gate_values.csv")
with open(gate_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genome_id", "gate_mean", "fusion_mode", "true_label", "prediction"])
    for i in range(n_test):
        mode = "KG-dominated" if gate_means[i] < 0.5 else "genomic-dominated"
        true_label = "RESISTANT" if labels[i] == 1 else "SUSCEPTIBLE"
        pred_label = "RESISTANT" if preds[i] == 1 else "SUSCEPTIBLE"
        writer.writerow([test_ids[i], f"{gate_means[i]:.6f}", mode, true_label, pred_label])
print(f"  Saved to fusion_gate_values.csv")

# ── 2C. SHAP via Gradient×Input Approximation (fast) ────────────────────────
print("\n[4/4] Computing SHAP via gradient×input approximation...")

from model.kg_amr_v2 import KGAMRv2

# Load features and model
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)

KMER_DIM = len(all_features)
NUM_GENES = len(gene_names)

# Load model
ckpt_path = os.path.join(MODEL_DIR, "checkpoints/best_model.ckpt")
model = KGAMRv2.load_from_checkpoint(ckpt_path, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
model.eval()

# Load X_test and gene_embeds_test
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_sample_indices = [sample_to_idx[s] for s in test_ids]
X_test = X_sparse[test_sample_indices].toarray().astype(np.float32)

# Load gene embeddings
entity_emb_raw = np.load(os.path.join(KG_DIR, "embeddings/entity_embeddings.npy"))
if np.iscomplexobj(entity_emb_raw):
    entity_emb = np.abs(entity_emb_raw).astype(np.float32)
else:
    entity_emb = entity_emb_raw.astype(np.float32)

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

# Compute SHAP via gradient·input on genomic encoder
print("  Computing gradient·input SHAP approximation...")
X_test_tensor = torch.tensor(X_test, requires_grad=True, dtype=torch.float32)
ge_test_tensor = torch.tensor(gene_embeds_test, dtype=torch.float32)

shap_values = np.zeros_like(X_test)

with torch.enable_grad():
    for i in range(n_test):
        if i % 1000 == 0:
            print(f"    [{i}/{n_test}]...")

        x_i = X_test_tensor[i:i+1]
        ge_i = ge_test_tensor[i:i+1]
        x_i.requires_grad_(True)

        # Forward pass
        logits, _, _, _ = model(x_i, ge_i)
        logit_r = logits[0, 1]  # logit for resistant class

        # Backward pass
        logit_r.backward()

        # SHAP ≈ gradient · input (DeepLIFT approximation)
        grad = x_i.grad.detach().numpy()[0]
        shap_values[i] = grad * X_test[i]

        x_i.requires_grad_(False)

print(f"  SHAP values computed: shape {shap_values.shape}")

# Save top-20 SHAP features per genome
shap_mean_abs = np.abs(shap_values).mean(axis=0)
shap_path = os.path.join(EXPLAIN_DIR, "shap_values.csv")
with open(shap_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genome_id", "rank", "feature", "gene", "shap_value", "card_mapped"])
    card_gene_set = set(gene_mechanism.keys())
    for i in range(n_test):
        genome_shap = np.abs(shap_values[i])
        genome_top = np.argsort(genome_shap)[::-1][:20]
        for rank, fi in enumerate(genome_top):
            feat = all_features[fi]
            gene = feat.split(":")[0]
            mapped = gene in card_gene_set
            writer.writerow([
                test_ids[i], rank + 1, feat, gene,
                f"{shap_values[i, fi]:.6f}",
                "yes" if mapped else "no",
            ])

print(f"  Saved top-20 SHAP per genome to shap_values.csv")

# Save raw SHAP for alignment metrics
np.savez_compressed(
    os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"),
    shap_values=shap_values,
    feature_names=np.array(all_features),
)

# Global top-20
top_idx = np.argsort(shap_mean_abs)[::-1][:20]
print("\n  Global top-20 SHAP features:")
for rank in range(20):
    fi = top_idx[rank]
    feat = all_features[fi]
    gene = feat.split(":")[0]
    mapped = "CARD" if gene in card_gene_set else "unmapped"
    print(f"    {rank+1:2d}. {feat:30s} [{mapped}]")

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.1f}s")
print("DONE — shap_analysis_optimized.py")
