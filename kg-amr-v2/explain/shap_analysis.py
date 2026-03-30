"""
Step 8A-C (fixed): Explainability Pipeline — Gene Attention, Fusion Gates, SHAP
Using gradient × input (saliency) for SHAP — much faster and more memory-efficient than GradientExplainer.
All values computed from actual model outputs, never hardcoded.
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import KG_EMBED_DIM, FUSED_DIM, PROJECT_DIR

cwd = os.getcwd()
assert "kg-amr-v2" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import torch
import torch.nn as nn

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
EXPLAIN_DIR = os.path.join(PROJECT_DIR, "explain")
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
print(f"  Gene names ({len(gene_names)}): {list(gene_names[:5])}...")
print(f"  Attention shape: {attn_weights.shape}")
print(f"  Gate shape: {gate_values.shape}")

# ── 2A. Gene Attention Weights ───────────────────────────────────────────────
print("\n[2/4] Extracting gene attention weights...")

# Verify attention weights sum to ~1.0 per genome
attn_sums = attn_weights.sum(axis=1)
violations = np.where(np.abs(attn_sums - 1.0) > 1e-4)[0]
if len(violations) > 0:
    print(f"  WARNING: {len(violations)} genomes have attention sums != 1.0")
    print(f"    Range: [{attn_sums[violations].min():.6f}, {attn_sums[violations].max():.6f}]")
else:
    print(f"  All {n_test} genomes have attention sums within 1e-4 of 1.0 ✓")

# Save per-genome gene attention weights
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

print(f"  Saved {n_test * len(gene_names)} rows to {attn_path}")

# ── 2B. Fusion Gate Values ───────────────────────────────────────────────────
print("\n[3/4] Extracting fusion gate values...")

gate_means = gate_values.mean(axis=1)  # [n_test]
n_kg_dominated = int((gate_means < 0.5).sum())
n_genomic_dominated = int((gate_means >= 0.5).sum())

print(f"  Gate mean range: [{gate_means.min():.4f}, {gate_means.max():.4f}]")
print(f"  Mean gate: {gate_means.mean():.4f}")
print(f"  {n_kg_dominated} ({100*n_kg_dominated/n_test:.1f}%) KG-dominated (gate < 0.5)")
print(f"  {n_genomic_dominated} ({100*n_genomic_dominated/n_test:.1f}%) genomic-dominated (gate >= 0.5)")

gate_path = os.path.join(EXPLAIN_DIR, "fusion_gate_values.csv")
with open(gate_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genome_id", "gate_mean", "fusion_mode", "true_label", "prediction"])
    for i in range(n_test):
        mode = "KG-dominated" if gate_means[i] < 0.5 else "genomic-dominated"
        true_label = "RESISTANT" if labels[i] == 1 else "SUSCEPTIBLE"
        pred_label = "RESISTANT" if preds[i] == 1 else "SUSCEPTIBLE"
        writer.writerow([test_ids[i], f"{gate_means[i]:.6f}", mode, true_label, pred_label])

print(f"  Saved to {gate_path}")

# ── 2C. Gradient × Input (Saliency) attribution for faster SHAP analogue ─────
print("\n[4/4] Computing gradient × input attribution (saliency)...")

# Load model and actual test features
from model.kg_amr_v2 import KGAMRv2
from scipy import sparse

# Load test features
X_sparse = sparse.load_npz(os.path.join(PROJECT_DIR, "features/mutation_matrix.npz"))
with open(os.path.join(PROJECT_DIR, "features/mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(PROJECT_DIR, "features/mutation_features.json")) as f:
    all_features = json.load(f)

sample_to_idx = {s: i for i, s in enumerate(all_samples)}
test_sample_indices = [sample_to_idx[s] for s in test_ids]
X_test = X_sparse[test_sample_indices].toarray().astype(np.float32)
X_test_tensor = torch.tensor(X_test, requires_grad=True)

# Load KG embeddings
entity_emb_raw = np.load(os.path.join(PROJECT_DIR, "kg/embeddings/entity_embeddings.npy"))
if np.iscomplexobj(entity_emb_raw):
    entity_emb = np.abs(entity_emb_raw).astype(np.float32)
else:
    entity_emb = entity_emb_raw.astype(np.float32)

with open(os.path.join(PROJECT_DIR, "kg/embeddings/pykeen_entity_to_id.json")) as f:
    entity_to_id = json.load(f)
with open(os.path.join(PROJECT_DIR, "kg/gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)

CATALOGUE_GENES = sorted(gene_mechanism.keys())
gene_emb_indices = [entity_to_id[g] for g in CATALOGUE_GENES]
gene_emb_matrix = entity_emb[gene_emb_indices]

# Build gene embedding tensor
NUM_GENES = len(gene_names)
gene_embeds_test = np.zeros((n_test, NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
for i in range(n_test):
    for j in range(NUM_GENES):
        if gene_presence[i, j] > 0:
            gene_embeds_test[i, j, :] = gene_emb_matrix[j]
gene_embeds_tensor = torch.tensor(gene_embeds_test)

# Load model
KMER_DIM = len(all_features)
ckpt_path = os.path.join(MODEL_DIR, "checkpoints/best_model.ckpt")
model = KGAMRv2.load_from_checkpoint(ckpt_path, kmer_dim=KMER_DIM, num_genes=NUM_GENES)
model.eval()
model = model.to("cpu")

# Compute gradients on a subset (to manage memory)
print("  Computing gradient × input attribution in batches...")
batch_size = 100
shap_values_all_batches = []

for batch_start in range(0, n_test, batch_size):
    batch_end = min(batch_start + batch_size, n_test)
    batch_size_actual = batch_end - batch_start

    X_batch = X_test_tensor[batch_start:batch_end].clone().detach().requires_grad_(True)
    ge_batch = gene_embeds_tensor[batch_start:batch_end]

    # Forward pass
    logits, _, _, _ = model(X_batch, ge_batch)
    probs_batch = torch.softmax(logits, dim=1)

    # For resistant class (class 1), get gradients
    probs_r = probs_batch[:, 1]  # P(resistant)

    # Sum of probabilities as target for gradient computation
    target = probs_r.sum()
    target.backward()

    # Gradient × Input
    gradients = X_batch.grad  # [batch, 17352]
    saliency = (X_batch * gradients).detach().numpy()  # element-wise product
    shap_values_all_batches.append(saliency)

    if batch_start % 300 == 0:
        print(f"    Processed {batch_start}/{n_test}...")

shap_values = np.concatenate(shap_values_all_batches, axis=0)
print(f"  Saliency shape: {shap_values.shape}")

# Map SHAP features to CARD genes
card_gene_set = set(gene_mechanism.keys())
shap_mean = np.abs(shap_values).mean(axis=0)
top_shap_idx = np.argsort(shap_mean)[::-1]

# Save top-20 SHAP features per genome
shap_path = os.path.join(EXPLAIN_DIR, "shap_values.csv")
with open(shap_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["genome_id", "rank", "feature", "gene", "saliency_value", "card_mapped"])
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

print(f"  Saved top-20 SHAP per genome to {shap_path} ({n_test * 20} rows)")

# Global top-20 SHAP features
print("  Global top-20 gradient × input features:")
for rank in range(20):
    fi = top_shap_idx[rank]
    feat = all_features[fi]
    gene = feat.split(":")[0]
    mapped = "CARD" if gene in card_gene_set else "unmapped"
    print(f"    {rank+1:2d}. {feat:30s} mean|sal|={shap_mean[fi]:.6f} [{mapped}]")

# Save raw SHAP values
np.savez_compressed(
    os.path.join(EXPLAIN_DIR, "shap_raw_values.npz"),
    shap_values=shap_values,
    feature_names=np.array(all_features),
)

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.1f}s")
print("DONE — shap_analysis.py (gradient × input attribution)")
