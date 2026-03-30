"""
Stage M6: Train KG-AMR v2 Per Species
Trains a separate KGAMRv2 model for each species using the phylogenetic CV folds.

CV fold assignment (10-fold phylogenetic):
  - Fold 9 (last)   → test
  - Fold 8           → validation
  - Folds 0-7       → training

Outputs per species:
  model/species/{species}/checkpoints/best.ckpt   (best val-F1 checkpoint)
  model/species/{species}/test_results.json        (F1, AUC, accuracy)
  model/species/{species}/test_outputs.npz         (probabilities + labels)
  model/species/{species}/training_log.csv
"""
import os, sys, json, time, csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from paths import PROJECT_DIR, KG_EMBED_DIM

# ── Paths ─────────────────────────────────────────────────────────────────
MATRIX_DIR = os.path.join(PROJECT_DIR, "features/matrices")
ANNOT_DIR  = os.path.join(PROJECT_DIR, "features/annotations")
LABELS_DIR = os.path.join(PROJECT_DIR, "features/labels")
KG_BASE    = os.path.join(PROJECT_DIR, "kg/species")
MODEL_BASE = os.path.join(PROJECT_DIR, "model/species")
CV_BASE    = os.path.expanduser("~/Desktop/AMR NamanXSarika/Mendeley Data/"
                                 "phylogenetic_trees_and_CV_folds/"
                                 "single_species_antibiotic_folds")
os.makedirs(MODEL_BASE, exist_ok=True)

# MPS check
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SPECIES_CONFIGS = {
    "Ecoli_ampicillin":       ("Escherichia_coli",        "ampicillin"),
    "Kpneumoniae_cipro":      ("Klebsiella_pneumoniae",   "ciprofloxacin"),
    "Kpneumoniae_carbapenem": ("Klebsiella_pneumoniae",   "meropenem"),
    "Abaumannii_carbapenem":  ("Acinetobacter_baumannii", "meropenem"),
}

# ── Dataset ─────────────────────────────────────────────────────────────────
class AMRDataset(Dataset):
    def __init__(self, X_kmer, gene_embeds, y_amr, y_genes):
        self.X_kmer      = torch.tensor(X_kmer, dtype=torch.float32)
        self.gene_embeds = torch.tensor(gene_embeds, dtype=torch.float32)
        self.y_amr       = torch.tensor(y_amr, dtype=torch.long)
        self.y_genes     = torch.tensor(y_genes, dtype=torch.float32)

    def __len__(self):
        return len(self.y_amr)

    def __getitem__(self, idx):
        return (self.X_kmer[idx], self.gene_embeds[idx],
                self.y_amr[idx], self.y_genes[idx])


# ── Model (inline so no path issues) ────────────────────────────────────────
from model.kg_amr_v2 import KGAMRv2


# ── Training helper ──────────────────────────────────────────────────────────
def train_species(species, cv_species_folder, cv_drug):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"Training: {species}")

    # ── Load k-mer features ──────────────────────────────────────────────
    kmer_npz  = os.path.join(MATRIX_DIR, f"{species}_kmer.npz")
    kmer_ids  = os.path.join(MATRIX_DIR, f"{species}_kmer_ids.npy")
    if not os.path.exists(kmer_npz):
        print(f"  [SKIP] Missing k-mer features: {kmer_npz}")
        return None

    X_sparse   = sp.load_npz(kmer_npz)
    genome_ids = np.load(kmer_ids, allow_pickle=True)
    KMER_DIM   = X_sparse.shape[1]
    print(f"  K-mer matrix: {X_sparse.shape}")

    # ── Load labels ───────────────────────────────────────────────────────
    labels_csv = os.path.join(LABELS_DIR, f"{species}_labels.csv")
    if not os.path.exists(labels_csv):
        print(f"  [SKIP] Missing labels: {labels_csv}")
        return None
    labels_df = pd.read_csv(labels_csv)
    labels_df["genome_id"] = labels_df["genome_id"].astype(str)
    label_map = dict(zip(labels_df["genome_id"], labels_df["resistant_phenotype"]))
    print(f"  Labels loaded: {len(label_map)}")

    # ── Load gene presence + KG embeddings ───────────────────────────────
    pres_npz    = os.path.join(ANNOT_DIR, f"{species}_gene_presence.npz")
    annot_ids   = np.load(os.path.join(ANNOT_DIR, f"{species}_gene_ids.npy"),
                          allow_pickle=True)
    kg_dir      = os.path.join(KG_BASE, species)
    genes_json  = os.path.join(ANNOT_DIR, f"{species}_genes.json")
    kg_e2id     = os.path.join(kg_dir, "entity_to_id.json")
    kg_emb_path = os.path.join(kg_dir, "entity_embeddings.npy")

    if os.path.exists(pres_npz) and os.path.exists(kg_emb_path):
        gene_pres_sparse = sp.load_npz(pres_npz)
        gene_pres = gene_pres_sparse.toarray().astype(np.float32)
        # annot_ids is indexed the same order as rows of gene_pres
        annot_id_list = list(annot_ids)

        with open(genes_json) as f:
            gene_to_col = json.load(f)
        genes_ordered = sorted(gene_to_col, key=gene_to_col.get)
        NUM_GENES = len(genes_ordered)

        with open(kg_e2id) as f:
            entity_to_id = json.load(f)
        entity_emb = np.load(kg_emb_path).astype(np.float32)

        # Build gene embedding matrix: [NUM_GENES, KG_EMBED_DIM]
        gene_emb_matrix = np.zeros((NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
        for gi, g in enumerate(genes_ordered):
            eid = entity_to_id.get(g)
            if eid is not None:
                gene_emb_matrix[gi] = entity_emb[eid]
    else:
        print(f"  [WARN] Missing annotations — using zero gene embeddings")
        NUM_GENES = 1
        genes_ordered = ["placeholder"]
        gene_pres = np.zeros((len(genome_ids), 1), dtype=np.float32)
        annot_id_list = list(genome_ids)
        gene_emb_matrix = np.zeros((1, KG_EMBED_DIM), dtype=np.float32)

    print(f"  NUM_GENES={NUM_GENES}, KG_EMBED_DIM={KG_EMBED_DIM}")

    # ── Load CV folds ─────────────────────────────────────────────────────
    fold_path = os.path.join(CV_BASE, cv_species_folder,
                             f"{cv_drug}_random_cv.json")
    if not os.path.exists(fold_path):
        print(f"  [WARN] CV fold file not found: {fold_path}")
        print(f"  Falling back to random 70/15/15 split")
        folds = None
    else:
        with open(fold_path) as f:
            folds = json.load(f)
        print(f"  CV folds loaded: {len(folds)} folds")

    # ── Align all arrays to intersection of kmer+labels+annotation IDs ───
    # genome_ids: from k-mer matrix
    # annot_id_list: from gene_pres matrix (same order as gene_pres rows)
    # label_map: genome_id → label
    annot_id_to_row = {gid: i for i, gid in enumerate(annot_id_list)}
    kmer_id_to_row  = {gid: i for i, gid in enumerate(genome_ids)}

    # Keep only genomes that have k-mer features AND labels
    valid_gids = [gid for gid in genome_ids
                  if gid in label_map]
    print(f"  Genomes with both k-mer features and labels: {len(valid_gids)}")

    if len(valid_gids) < 50:
        print(f"  [SKIP] Too few labelled genomes ({len(valid_gids)})")
        return None

    # Build aligned arrays
    kmer_rows  = [kmer_id_to_row[gid] for gid in valid_gids]
    annot_rows = [annot_id_to_row.get(gid, -1) for gid in valid_gids]

    X_kmer_all = X_sparse[kmer_rows].toarray().astype(np.float32)
    y_all      = np.array([label_map[gid] for gid in valid_gids], dtype=np.int64)

    # Per-genome gene embeddings: [N, NUM_GENES, KG_EMBED_DIM]
    gene_embeds_all = np.zeros((len(valid_gids), NUM_GENES, KG_EMBED_DIM), dtype=np.float32)
    y_genes_all     = np.zeros((len(valid_gids), NUM_GENES), dtype=np.float32)
    for i, (gid, arow) in enumerate(zip(valid_gids, annot_rows)):
        if arow >= 0 and arow < len(gene_pres):
            pres_vec = gene_pres[arow]  # [NUM_GENES] binary
            y_genes_all[i] = pres_vec
            for j in range(NUM_GENES):
                if pres_vec[j] > 0:
                    gene_embeds_all[i, j] = gene_emb_matrix[j]

    N = len(valid_gids)
    print(f"  Final dataset: {N} samples (k-mer dim={KMER_DIM})")
    print(f"  Label dist: {(y_all==1).sum()} R / {(y_all==0).sum()} S "
          f"({100*(y_all==1).mean():.1f}% R)")

    # ── Build train/val/test masks ─────────────────────────────────────────
    valid_gid_set = set(valid_gids)

    if folds is not None:
        # Use last fold as test, second-to-last as val, rest as train
        test_ids = set(str(g) for g in folds[-1]) & valid_gid_set
        val_ids  = set(str(g) for g in folds[-2]) & valid_gid_set
        train_ids = set()
        for fold in folds[:-2]:
            train_ids |= (set(str(g) for g in fold) & valid_gid_set)

        # Any valid genomes not in any fold → add to train
        assigned = test_ids | val_ids | train_ids
        unassigned = valid_gid_set - assigned
        train_ids |= unassigned

        gid_list = list(valid_gids)
        idx_train = [i for i, g in enumerate(gid_list) if g in train_ids]
        idx_val   = [i for i, g in enumerate(gid_list) if g in val_ids]
        idx_test  = [i for i, g in enumerate(gid_list) if g in test_ids]

        # Fallback if fold coverage is poor
        if len(idx_test) < 10 or len(idx_val) < 10:
            print(f"  [WARN] Poor fold coverage (test={len(idx_test)}, "
                  f"val={len(idx_val)}) — using random 70/15/15 split")
            folds = None
    
    if folds is None:
        from sklearn.model_selection import train_test_split
        indices = np.arange(N)
        idx_tr, idx_tmp = train_test_split(indices, test_size=0.30,
                                           random_state=42, stratify=y_all)
        idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50,
                                             random_state=42, stratify=y_all[idx_tmp])
        idx_train = list(idx_tr)

    idx_train, idx_val, idx_test = list(idx_train), list(idx_val), list(idx_test)
    print(f"  Split — train:{len(idx_train)}  val:{len(idx_val)}  test:{len(idx_test)}")

    # Check for empty split
    if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
        print(f"  [SKIP] Empty split — cannot train")
        return None

    # ── DataLoaders ──────────────────────────────────────────────────────
    def make_loader(indices, shuffle):
        idx = np.array(indices)
        ds = AMRDataset(
            X_kmer_all[idx],
            gene_embeds_all[idx],
            y_all[idx],
            y_genes_all[idx],
        )
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader   = make_loader(idx_val,   shuffle=False)
    test_loader  = make_loader(idx_test,  shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = KGAMRv2(kmer_dim=KMER_DIM, num_genes=NUM_GENES)

    sp_model_dir = os.path.join(MODEL_BASE, species)
    ckpt_dir     = os.path.join(sp_model_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_csv_path = os.path.join(sp_model_dir, "training_log.csv")

    callbacks = [
        EarlyStopping(monitor="val_f1_macro", mode="max",
                      patience=12, verbose=True),
        ModelCheckpoint(dirpath=ckpt_dir, filename="best",
                        monitor="val_f1_macro", mode="max",
                        save_top_k=1, verbose=False),
    ]

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator=DEVICE,
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
        logger=pl.loggers.CSVLogger(sp_model_dir, name="", version=""),
    )

    print(f"  Training on {DEVICE}...")
    trainer.fit(model, train_loader, val_loader)

    # ── Test evaluation ────────────────────────────────────────────────────
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
    if os.path.exists(best_ckpt):
        model = KGAMRv2.load_from_checkpoint(best_ckpt)
    eval_device = torch.device(DEVICE)
    model = model.to(eval_device)
    model.eval()

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for kmer_x, ge, y_amr, y_genes in test_loader:
            kmer_x = kmer_x.to(eval_device)
            ge     = ge.to(eval_device)
            y_amr  = y_amr.to(eval_device)
            logits, _, _, _ = model(kmer_x, ge)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_targets.extend(y_amr.cpu().numpy().tolist())

    all_probs   = np.array(all_probs)
    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_R     = f1_score(all_targets, all_preds, pos_label=1, average="binary",
                        zero_division=0)
    f1_S     = f1_score(all_targets, all_preds, pos_label=0, average="binary",
                        zero_division=0)
    acc      = accuracy_score(all_targets, all_preds)
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = float("nan")

    elapsed = time.time() - t0
    print(f"\n  Test results for {species}:")
    print(f"    F1-macro  : {f1_macro:.4f}")
    print(f"    F1-R      : {f1_R:.4f}")
    print(f"    F1-S      : {f1_S:.4f}")
    print(f"    AUC-ROC   : {auc:.4f}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Time      : {elapsed:.0f}s")

    results = {
        "species": species,
        "n_train": len(idx_train),
        "n_val":   len(idx_val),
        "n_test":  len(idx_test),
        "f1_macro": round(f1_macro, 4),
        "f1_R":     round(f1_R, 4),
        "f1_S":     round(f1_S, 4),
        "auc_roc":  round(float(auc), 4) if not np.isnan(auc) else None,
        "accuracy": round(acc, 4),
        "kmer_dim": int(KMER_DIM),
        "num_genes": int(NUM_GENES),
        "elapsed_s": round(elapsed, 1),
    }

    with open(os.path.join(sp_model_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(sp_model_dir, "test_outputs.npz"),
             probs=all_probs, preds=all_preds, targets=all_targets,
             genome_ids=np.array([valid_gids[i] for i in idx_test]))

    return results


# ── Main ────────────────────────────────────────────────────────────────────
all_results = []
for species, (cv_folder, cv_drug) in SPECIES_CONFIGS.items():
    r = train_species(species, cv_folder, cv_drug)
    if r:
        all_results.append(r)

print("\n\n=== Multi-Species Results Summary ===")
print(f"  {'Species':<28}  {'F1-macro':>8}  {'AUC-ROC':>8}  {'Acc':>7}  {'Test N':>7}")
print(f"  {'-'*65}")
for r in all_results:
    auc_str = f"{r['auc_roc']:.4f}" if r["auc_roc"] is not None else "   N/A"
    print(f"  {r['species']:<28}  {r['f1_macro']:>8.4f}  "
          f"{auc_str:>8}  {r['accuracy']:>7.4f}  {r['n_test']:>7}")

print("\nStage M6 complete.")
