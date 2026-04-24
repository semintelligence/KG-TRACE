"""
Stage M7: Per-Species Explainability
Computes SHAP values and attention weights for each species model.
Identifies top genomic k-mer features and top resistance genes per species.

Outputs per species:
  explain/species/{species}/shap_values.csv         (top k-mer features by SHAP)
  explain/species/{species}/gene_attention.csv       (gene attention weights)
  explain/species/{species}/gate_values.csv          (fusion gate statistics)
  explain/species/{species}/figures/                 (HTML + PNG figures)
"""
import os, sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR, KG_EMBED_DIM

MATRIX_DIR = os.path.join(PROJECT_DIR, "features/matrices")
ANNOT_DIR  = os.path.join(PROJECT_DIR, "features/annotations")
LABELS_DIR = os.path.join(PROJECT_DIR, "features/labels")
KG_BASE    = os.path.join(PROJECT_DIR, "kg/species")
MODEL_BASE = os.path.join(PROJECT_DIR, "model/species")
EXPLAIN_BASE = os.path.join(PROJECT_DIR, "explain/species")
os.makedirs(EXPLAIN_BASE, exist_ok=True)

SPECIES_LIST = [
    "Ecoli_ampicillin",
    "Kpneumoniae_cipro",
    "Kpneumoniae_carbapenem",
    "Abaumannii_carbapenem",
]

SPECIES_NICE = {
    "Ecoli_ampicillin":       "E. coli (Ampicillin)",
    "Kpneumoniae_cipro":      "K. pneumoniae (Ciprofloxacin)",
    "Kpneumoniae_carbapenem": "K. pneumoniae (Carbapenem)",
    "Abaumannii_carbapenem":  "A. baumannii (Carbapenem)",
}

SHAP_BACKGROUND = 100  # background samples for SHAP estimation
TOP_K_FEATURES  = 25   # top k-mer features to report


def load_species_data(species):
    """Load all data arrays for a species. Returns dict or None on failure."""
    kmer_npz = os.path.join(MATRIX_DIR, f"{species}_kmer.npz")
    kmer_ids = os.path.join(MATRIX_DIR, f"{species}_kmer_ids.npy")
    if not os.path.exists(kmer_npz):
        return None

    X_sparse   = sp.load_npz(kmer_npz)
    genome_ids = np.load(kmer_ids, allow_pickle=True)

    labels_csv = os.path.join(LABELS_DIR, f"{species}_labels.csv")
    if not os.path.exists(labels_csv):
        return None
    labels_df  = pd.read_csv(labels_csv)
    labels_df["genome_id"] = labels_df["genome_id"].astype(str)
    label_map = dict(zip(labels_df["genome_id"], labels_df["resistant_phenotype"]))

    # Align
    valid_gids = [g for g in genome_ids if g in label_map]
    kmer_row = {g: i for i, g in enumerate(genome_ids)}
    row_idx  = [kmer_row[g] for g in valid_gids]

    X = X_sparse[row_idx].toarray().astype(np.float32)
    y = np.array([label_map[g] for g in valid_gids], dtype=np.int64)

    # Load vocab
    vocab_file = os.path.join(MATRIX_DIR, f"{species}_kmer_vocab.json")
    if os.path.exists(vocab_file):
        with open(vocab_file) as f:
            vocab = json.load(f)
        idx_to_kmer = {v: k for k, v in vocab.items()}
    else:
        idx_to_kmer = {}

    # Load gene data
    pres_npz  = os.path.join(ANNOT_DIR, f"{species}_gene_presence.npz")
    annot_ids = os.path.join(ANNOT_DIR, f"{species}_gene_ids.npy")
    genes_js  = os.path.join(ANNOT_DIR, f"{species}_genes.json")
    kg_dir    = os.path.join(KG_BASE, species)
    ke2id     = os.path.join(kg_dir, "entity_to_id.json")
    kemb      = os.path.join(kg_dir, "entity_embeddings.npy")

    if os.path.exists(pres_npz) and os.path.exists(kemb):
        gene_pres_sparse = sp.load_npz(pres_npz)
        gene_pres     = gene_pres_sparse.toarray().astype(np.float32)
        annot_id_list = list(np.load(annot_ids, allow_pickle=True))
        with open(genes_js) as f:
            gene_to_col  = json.load(f)
        genes_ordered = sorted(gene_to_col, key=gene_to_col.get)
        with open(ke2id) as f:
            entity_to_id = json.load(f)
        entity_emb   = np.load(kemb).astype(np.float32)

        gene_emb_matrix = np.zeros((len(genes_ordered), KG_EMBED_DIM), dtype=np.float32)
        for gi, g in enumerate(genes_ordered):
            eid = entity_to_id.get(g)
            if eid is not None:
                gene_emb_matrix[gi] = entity_emb[eid]

        annot_id_to_row = {g: i for i, g in enumerate(annot_id_list)}
        NUM_GENES = len(genes_ordered)
    else:
        NUM_GENES       = 1
        genes_ordered  = ["placeholder"]
        gene_pres      = np.zeros((len(valid_gids), 1), dtype=np.float32)
        annot_id_to_row= {g: i for i, g in enumerate(valid_gids)}
        gene_emb_matrix= np.zeros((1, KG_EMBED_DIM), dtype=np.float32)

    gene_embeds_all = np.zeros((len(valid_gids), NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
    y_genes_all     = np.zeros((len(valid_gids), NUM_GENES), dtype=np.float32)
    for i, gid in enumerate(valid_gids):
        arow = annot_id_to_row.get(gid, -1)
        if arow >= 0 and arow < len(gene_pres):
            pv = gene_pres[arow]
            y_genes_all[i] = pv
            for j in range(NUM_GENES):
                if pv[j] > 0:
                    gene_embeds_all[i, j] = gene_emb_matrix[j]

    return {
        "X": X, "y": y, "valid_gids": valid_gids,
        "gene_embeds": gene_embeds_all, "y_genes": y_genes_all,
        "genes_ordered": genes_ordered, "idx_to_kmer": idx_to_kmer,
        "NUM_GENES": NUM_GENES,
    }


from model.kg_trace import KGTrace

for species in SPECIES_LIST:
    print(f"\n{'='*60}")
    print(f"Explaining: {species}")

    sp_model_dir = os.path.join(MODEL_BASE, species)
    best_ckpt    = os.path.join(sp_model_dir, "checkpoints", "best.ckpt")
    sp_explain   = os.path.join(EXPLAIN_BASE, species)
    os.makedirs(os.path.join(sp_explain, "figures"), exist_ok=True)

    if not os.path.exists(best_ckpt):
        print(f"  [SKIP] No checkpoint found: {best_ckpt}")
        continue

    data = load_species_data(species)
    if data is None:
        print(f"  [SKIP] Cannot load data")
        continue

    X = data["X"]
    y = data["y"]
    NUM_GENES    = data["NUM_GENES"]
    gene_embeds  = data["gene_embeds"]
    genes_ordered= data["genes_ordered"]
    idx_to_kmer  = data["idx_to_kmer"]

    model = KGTrace.load_from_checkpoint(best_ckpt)
    # Use CPU for gradient-based analysis (MPS doesn't support grad on inputs reliably)
    eval_device = torch.device("cpu")
    model = model.to(eval_device)
    model.eval()

    N_explain = min(500, len(X))
    rng = np.random.default_rng(42)
    explain_idx = rng.choice(len(X), size=N_explain, replace=False)

    # ── 1. Attention weights + gate values ────────────────────────────────
    attn_list, gate_list = [], []
    batch_size = 64
    with torch.no_grad():
        for start in range(0, N_explain, batch_size):
            end     = min(start + batch_size, N_explain)
            bidx    = explain_idx[start:end]
            kx      = torch.tensor(X[bidx], dtype=torch.float32).to(eval_device)
            ge      = torch.tensor(gene_embeds[bidx], dtype=torch.float32).to(eval_device)
            _, _, attn, gate = model(kx, ge)
            attn_list.append(attn.cpu().numpy())
            gate_list.append(gate.cpu().numpy())

    # Attention has shape [batch, NUM_GENES]
    attn_all = np.concatenate(attn_list, axis=0)   # [N_explain, NUM_GENES]
    gate_all = np.concatenate(gate_list, axis=0)    # [N_explain, FUSED_DIM]

    # Per-gene mean attention weight
    mean_attn = attn_all.mean(axis=0)
    gene_attn_df = pd.DataFrame({
        "gene":        genes_ordered,
        "mean_attn":   mean_attn,
    }).sort_values("mean_attn", ascending=False)
    gene_attn_df.to_csv(os.path.join(sp_explain, "gene_attention.csv"), index=False)

    # Gate statistics
    mean_gate = gate_all.mean(axis=0)
    gate_df = pd.DataFrame({
        "gate_dim_idx": np.arange(len(mean_gate)),
        "mean_gate":    mean_gate,
    })
    gate_df.to_csv(os.path.join(sp_explain, "gate_values.csv"), index=False)
    print(f"  Mean gate value: {mean_gate.mean():.4f} "
          f"(high→genomic, low→KG)")

    # ── 2. Gradient-based feature importance for k-mer features ──────────
    print(f"  Computing gradient importance for {N_explain} samples...")
    grad_accum = np.zeros(X.shape[1], dtype=np.float64)

    for start in range(0, N_explain, batch_size):
        end  = min(start + batch_size, N_explain)
        bidx = explain_idx[start:end]
        kx   = torch.tensor(X[bidx], dtype=torch.float32,
                             requires_grad=True).to(eval_device)
        ge   = torch.tensor(gene_embeds[bidx], dtype=torch.float32).to(eval_device)
        logits, _, _, _ = model(kx, ge)
        # Gradient of resistant-class logit w.r.t. k-mer input
        score = logits[:, 1].sum()
        score.backward()
        grad_accum += np.abs(kx.grad.detach().cpu().numpy()).mean(axis=0)

    grad_importance = grad_accum / (N_explain / batch_size)
    top_feat_idx = np.argsort(grad_importance)[::-1][:TOP_K_FEATURES]

    shap_df = pd.DataFrame({
        "feature_idx": top_feat_idx,
        "kmer":        [idx_to_kmer.get(int(i), f"kmer_{i}")
                        for i in top_feat_idx],
        "importance":  grad_importance[top_feat_idx],
    })
    shap_df.to_csv(os.path.join(sp_explain, "shap_values.csv"), index=False)
    print(f"  Top k-mer feature: {shap_df.iloc[0]['kmer']}")

    # ── 3. Figures ────────────────────────────────────────────────────────
    nice_name = SPECIES_NICE.get(species, species)

    # Fig A: Top k-mer feature importances
    fig_kmer = px.bar(
        shap_df.head(20), x="importance", y="kmer",
        orientation="h",
        title=f"Top 20 K-mer Features — {nice_name}",
        labels={"importance": "Mean |Gradient|", "kmer": "21-mer"},
        color="importance", color_continuous_scale="Blues",
    )
    fig_kmer.update_layout(yaxis={"autorange": "reversed"}, height=600)
    fig_kmer.write_html(os.path.join(sp_explain, "figures", "kmer_importance.html"))
    try:
        fig_kmer.write_image(os.path.join(sp_explain, "figures", "kmer_importance.png"),
                             scale=2, width=900, height=600)
    except Exception:
        pass

    # Fig B: Gene attention weights (top 20)
    top_genes = gene_attn_df.head(20)
    fig_gene = px.bar(
        top_genes, x="mean_attn", y="gene",
        orientation="h",
        title=f"Top 20 Gene Attention Weights — {nice_name}",
        labels={"mean_attn": "Mean Attention Weight", "gene": "Resistance Gene"},
        color="mean_attn", color_continuous_scale="Reds",
    )
    fig_gene.update_layout(yaxis={"autorange": "reversed"}, height=600)
    fig_gene.write_html(os.path.join(sp_explain, "figures", "gene_attention.html"))
    try:
        fig_gene.write_image(os.path.join(sp_explain, "figures", "gene_attention.png"),
                             scale=2, width=900, height=600)
    except Exception:
        pass

    # Fig C: Gate distribution
    fig_gate = px.histogram(
        x=gate_all.mean(axis=1),
        nbins=40,
        title=f"Fusion Gate Distribution — {nice_name}",
        labels={"x": "Mean Gate Value (0=KG, 1=Genomic)"},
        color_discrete_sequence=["#2ca02c"],
    )
    fig_gate.write_html(os.path.join(sp_explain, "figures", "gate_distribution.html"))
    try:
        fig_gate.write_image(os.path.join(sp_explain, "figures", "gate_distribution.png"),
                             scale=2, width=700, height=400)
    except Exception:
        pass

    print(f"  Figures saved to: {sp_explain}/figures/")

print("\nStage M7 complete.")
