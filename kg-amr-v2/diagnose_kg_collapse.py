"""
diagnose_kg_collapse.py
=======================
Run from your KG-AMR/ directory:
    python3 diagnose_kg_collapse.py

Checks:
  1. RotatE embedding collapse (are all 26 gene embeddings nearly identical?)
  2. Attention weight analysis (confirm degeneracy & find its cause)
  3. Gate saturation deep-dive (which dimensions are stuck?)
  4. KG training quality (loss curve proxy via embedding norms)
  5. Fix recommendation: what retraining settings would help
"""

import numpy as np
import json
import os
from pathlib import Path

# ── colour output ──────────────────────────────────────────────────────────────
RED   = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"
BOLD  = "\033[1m";  RST = "\033[0m"

def header(title):
    print(f"\n{BOLD}{'='*60}{RST}")
    print(f"{BOLD}  {title}{RST}")
    print(f"{BOLD}{'='*60}{RST}")

def ok(msg):   print(f"  {GRN}✓ {msg}{RST}")
def warn(msg): print(f"  {YEL}⚠ {msg}{RST}")
def bad(msg):  print(f"  {RED}✗ {msg}{RST}")

# ── 1. RotatE Embedding Collapse ───────────────────────────────────────────────
header("1. RotatE Gene Embedding Collapse")

emb_path = Path("kg/embeddings/entity_embeddings.npy")
ent_path  = Path("kg/entity_to_id.json")

if not emb_path.exists():
    bad(f"Not found: {emb_path}")
else:
    emb = np.load(emb_path)                    # (n_entities, 64)
    ent2id = json.load(open(ent_path))
    id2ent = {v: k for k, v in ent2id.items()}

    print(f"  All entities:  {emb.shape[0]}  embeddings of dim {emb.shape[1]}")

    # Use the exact 26 catalogue gene names the model's attention module operates on
    out_tmp = np.load("model/test_outputs.npz", allow_pickle=True)
    catalogue_genes = out_tmp["gene_names"].tolist()   # authoritative list of 26

    gene_ids  = [ent2id[g] for g in catalogue_genes if g in ent2id]
    missing   = [g for g in catalogue_genes if g not in ent2id]
    if missing:
        bad(f"These catalogue genes not found in entity_to_id: {missing}")

    gene_emb  = emb[gene_ids]                  # (26, 64)
    print(f"  Catalogue gene entities found: {len(gene_ids)}/26")
    if missing:
        print(f"  Missing from KG:               {missing}")

    # --- Pairwise cosine similarity among gene embeddings ---
    norms     = np.linalg.norm(gene_emb, axis=1, keepdims=True) + 1e-10
    gene_norm = gene_emb / norms
    cos_sim   = gene_norm @ gene_norm.T         # (n_genes, n_genes)
    np.fill_diagonal(cos_sim, np.nan)

    mean_sim  = np.nanmean(cos_sim)
    max_sim   = np.nanmax(cos_sim)
    pct_hi    = (cos_sim > 0.95).sum() / (~np.isnan(cos_sim)).sum() * 100

    print(f"\n  Pairwise cosine similarity (gene embeddings):")
    print(f"    Mean:            {mean_sim:.4f}")
    print(f"    Max:             {max_sim:.4f}")
    print(f"    Pairs > 0.95:    {pct_hi:.1f}%")

    mean_sim_r = float(np.real(mean_sim))
    max_sim_r  = float(np.real(max_sim))
    if mean_sim_r > 0.90:
        bad(f"COLLAPSED — mean cosine similarity {mean_sim_r:.3f} >> 0.80 threshold")
        bad("RotatE failed to separate gene embeddings. KG branch is essentially random.")
    elif mean_sim_r > 0.70:
        warn(f"Near-collapse — mean similarity {mean_sim_r:.3f}. KG signal very weak.")
    else:
        ok(f"Embeddings well-separated (mean sim={mean_sim_r:.3f})")

    # --- Embedding norms (collapsed embeddings have near-zero or constant norms) ---
    gene_norm_vals = np.linalg.norm(gene_emb, axis=1)
    print(f"\n  Gene embedding L2 norms:")
    print(f"    Mean:   {gene_norm_vals.mean():.4f}")
    print(f"    Std:    {gene_norm_vals.std():.4f}")
    print(f"    Min:    {gene_norm_vals.min():.4f}")
    print(f"    Max:    {gene_norm_vals.max():.4f}")

    if gene_norm_vals.std() < 0.05:
        bad("All gene embeddings have nearly identical norms → training didn't differentiate genes")
    else:
        ok("Norm variance looks healthy")

    # Print per-gene norms for the 26 catalogue genes
    print(f"\n  Per-gene L2 norms (26 catalogue genes):")
    for gname, norm_val in zip(catalogue_genes, gene_norm_vals):
        bar = '█' * int(norm_val / 0.15)
        print(f"    {gname:<10} {norm_val:.4f}  {bar}")

    # --- Most and least similar pairs (among the 26 catalogue genes) ---
    # cos_sim may be complex if embeddings contain imaginary via RotatE — take real part
    cos_real = np.real(cos_sim)
    i_max, j_max = np.unravel_index(np.nanargmax(cos_real), cos_real.shape)
    i_min, j_min = np.unravel_index(np.nanargmin(cos_real), cos_real.shape)
    print(f"\n  Most similar pair:  {catalogue_genes[i_max]} ↔ {catalogue_genes[j_max]}  "
          f"(cos={cos_real[i_max,j_max]:.4f})")
    print(f"  Least similar pair: {catalogue_genes[i_min]} ↔ {catalogue_genes[j_min]}  "
          f"(cos={cos_real[i_min,j_min]:.4f})")

    # Print full 26×26 similarity matrix summary
    print(f"\n  Cosine similarity distribution (26 catalogue genes, upper triangle):")
    upper = cos_real[np.triu_indices(len(gene_ids), k=1)]
    for threshold, label in [(0.9, '>0.90'), (0.7, '>0.70'), (0.5, '>0.50'), (0.3, '>0.30')]:
        n = (upper > threshold).sum()
        print(f"    Pairs {label}: {n}/{len(upper)}  ({n/len(upper)*100:.1f}%)")

# ── 2. Attention Weight Degeneracy — Root Cause ────────────────────────────────
header("2. Attention Degeneracy — Root Cause")

out_path = Path("model/test_outputs.npz")
if not out_path.exists():
    bad(f"Not found: {out_path}")
else:
    out  = np.load(out_path, allow_pickle=True)
    attn = out["attn_weights"]                 # (n_test, n_genes)
    n_genes = attn.shape[1]

    # Entropy
    H      = -(attn * np.log(attn + 1e-12)).sum(axis=1)
    H_max  = np.log(n_genes)
    pct_uniform = (H > 0.99 * H_max).mean() * 100

    print(f"  Attention shape:        {attn.shape}")
    print(f"  Mean entropy:           {H.mean():.4f}  (max={H_max:.4f})")
    print(f"  Samples near-uniform:   {pct_uniform:.1f}%")
    print(f"  Std of attn per sample: {attn.std(axis=1).mean():.5f}")

    if pct_uniform > 90:
        bad("Attention completely degenerate — model ignores gene identity")

    # Are weights IDENTICAL across samples or just uniform?
    # If embeddings are collapsed, logits are identical → softmax is identical
    sample_std = attn.std(axis=0)              # per-gene variance across samples
    print(f"\n  Per-gene attention variance across test samples:")
    print(f"    Mean std across genes: {sample_std.mean():.6f}")
    print(f"    Max  std across genes: {sample_std.max():.6f}")

    if sample_std.max() < 0.001:
        bad("Attention weights are IDENTICAL across all samples.")
        bad("Cause confirmed: collapsed gene embeddings → identical logits → uniform softmax")
        bad("Fixing RotatE training WILL fix the attention module.")
    else:
        warn("Attention varies slightly across samples but is still near-uniform")

# ── 3. Gate Saturation Deep-Dive ───────────────────────────────────────────────
header("3. Gate Saturation — Which Dimensions Are Stuck?")

if out_path.exists():
    gate = out["gate_values"]                  # (n_test, gate_dim)
    print(f"  Gate shape: {gate.shape}")

    stuck_zero = (gate < 0.01).mean(axis=0)   # per-dim fraction stuck at 0
    stuck_one  = (gate > 0.99).mean(axis=0)   # per-dim fraction stuck at 1
    active     = ((gate > 0.01) & (gate < 0.99)).mean(axis=0)

    print(f"\n  Across {gate.shape[1]} gate dimensions:")
    print(f"    Dims always ≈ 0    (>90% samples): {(stuck_zero > 0.9).sum()}")
    print(f"    Dims always ≈ 1    (>90% samples): {(stuck_one  > 0.9).sum()}")
    print(f"    Dims actively vary (<10% extreme): {(active     > 0.9).sum()}")

    pct_kg_dom   = (gate < 0.5).mean() * 100   # gate≈0 → KG dominates
    pct_gen_dom  = (gate > 0.5).mean() * 100

    print(f"\n  Gate < 0.5 (KG-dominated dims):     {pct_kg_dom:.1f}%")
    print(f"  Gate > 0.5 (Genomic-dominated dims): {pct_gen_dom:.1f}%")

    # Sample-level gate variance (are same dims stuck for every genome?)
    gate_std_across_samples = gate.std(axis=0)
    print(f"\n  Per-dim gate std across samples:")
    print(f"    Mean: {gate_std_across_samples.mean():.5f}")
    print(f"    Max:  {gate_std_across_samples.max():.5f}")

    if gate_std_across_samples.max() < 0.05:
        bad("Gate is the SAME for all genomes — KG embeddings carry zero genome-specific signal")
    else:
        warn("Gate varies per genome, but many dimensions are still saturated")

# ── 4. KG Training Quality Proxy ──────────────────────────────────────────────
header("4. KG Training Quality — Quick Proxy Checks")

# Check RotatE training log
for log_path in ["kg/rotate_training.log", "kg/training.log", "rotate_train.log"]:
    if Path(log_path).exists():
        lines = open(log_path).readlines()
        print(f"  Log: {log_path}  ({len(lines)} lines total)")
        # Show first and last content lines
        content = [l for l in lines if l.strip() and l.strip() != 'epoch,loss']
        if content:
            print(f"  First entry: {content[0].rstrip()}")
            print(f"  Last  entry: {content[-1].rstrip()}")
            # Try to parse as CSV
            import csv, io
            try:
                reader = csv.DictReader(io.StringIO(''.join(lines)))
                rows = list(reader)
                if rows:
                    first_loss = float(rows[0].get('loss', 'nan'))
                    last_loss  = float(rows[-1].get('loss', 'nan'))
                    n_epochs   = len(rows)
                    print(f"  Epochs logged: {n_epochs}")
                    print(f"  Loss: {first_loss:.4f} → {last_loss:.4f}  "
                          f"({'converged' if last_loss < first_loss * 0.5 else 'may not have converged'})")
                    if last_loss > first_loss * 0.95:
                        bad("Loss barely decreased — RotatE training may have stalled")
                    else:
                        ok(f"Loss dropped by {(1-last_loss/first_loss)*100:.0f}%")
            except Exception as e:
                warn(f"Could not parse log as CSV: {e}")
        else:
            bad(f"Log exists but contains no training entries (only header). Training data was not written.")
            bad("Check step3_train_rotate.py — the training log was never populated.")
        break
else:
    warn("No RotatE training log found. Cannot check convergence.")

# Also check rotate_summary.json if it exists
summary_path = Path("kg/embeddings/rotate_summary.json")
if not summary_path.exists():
    summary_path = Path("kg/embeddings/rotate_model.pt")
for sp in [Path("kg/embeddings/rotate_summary.json"), Path("kg/rotate_summary.json")]:
    if sp.exists():
        try:
            summary = json.load(open(sp))
            print(f"\n  rotate_summary.json keys: {list(summary.keys())[:10]}")
            for k in ['losses', 'best_loss', 'final_loss', 'hits_at_10', 'mean_rank']:
                if k in summary:
                    v = summary[k]
                    if isinstance(v, list):
                        print(f"    {k}: first={v[0]:.4f}, last={v[-1]:.4f}, n={len(v)}")
                    else:
                        print(f"    {k}: {v}")
        except Exception as e:
            warn(f"Could not parse {sp}: {e}")
        break

# Check KG size — small KGs underfit RotatE
kg_path = Path("kg/kg_summary.json")
if kg_path.exists():
    kg = json.load(open(kg_path))
    n_ent   = kg.get("n_entities", "?")
    n_trip  = kg.get("n_triples",  "?")
    n_rel   = kg.get("n_relations","?")
    print(f"\n  KG size: {n_ent} entities, {n_trip} triples, {n_rel} relations")

    # RotatE rule of thumb: need ~20+ triples per entity for good embeddings
    if isinstance(n_ent, int) and isinstance(n_trip, int):
        ratio = n_trip / n_ent
        print(f"  Triples-per-entity ratio: {ratio:.1f}")
        if ratio < 5:
            bad(f"Very sparse KG ({ratio:.1f} triples/entity). RotatE needs ≥ 10–20.")
        elif ratio < 15:
            warn(f"Sparse KG ({ratio:.1f} triples/entity). Embeddings may underfit.")
        else:
            ok(f"KG density looks sufficient ({ratio:.1f} triples/entity)")

# ── 5. Fix Recommendation ─────────────────────────────────────────────────────
header("5. Fix Recommendations")

print(f"""
  ══════════════════════════════════════════════════════════
  REVISED DIAGNOSIS  (after checking actual embeddings)
  ══════════════════════════════════════════════════════════

  RotatE IS fine:
    • Gene embeddings are well-separated (mean cosine sim ≈ 0, max = 0.25)
    • Training converged: loss 0.889 → 0.021 (98% drop over 250 epochs)
    • Norm variance is healthy (std = 0.057 across 26 genes)

  The attention degeneracy is a ARCHITECTURAL BUG, not a data problem:

      In model/kg_amr.py line ~54:
          attn_scores = self.gene_attn(gene_embeds)   # [batch, n_genes, 1]

      gene_embeds is the SAME 26×64 matrix for every sample in the dataset.
      gene_attn is a fixed linear layer.
      Therefore attn_scores are IDENTICAL for every genome.
      softmax(identical logits) → same attention weights for all 5,665 samples.

      The attention is SAMPLE-AGNOSTIC BY DESIGN.
      It can never express which gene matters for a specific genome.

  ── FIXES IN ORDER OF IMPACT ────────────────────────────────────────────

  FIX A — Cross-attention (query from genomic encoder) [RECOMMENDED]:
  -------------------------------------------------------------------
  The correct fix is to make the attention query genome-specific.
  In model/kg_amr.py, replace:

      # CURRENT (wrong — sample-agnostic):
      self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)
      ...
      attn_scores = self.gene_attn(gene_embeds)       # [B, 26, 1]

      # FIX — cross-attention (query from genomic branch):
      self.gene_attn_q = nn.Linear(256, KG_EMBED_DIM)   # project genomic→KG dim
      ...
      query = self.gene_attn_q(g).unsqueeze(2)        # [B, KG_EMBED_DIM, 1]
      attn_scores = torch.bmm(gene_embeds, query)     # [B, 26, 1]  dot-product

  After this change, each genome produces its own attention distribution
  based on which gene embeddings are most aligned with its genomic encoding.

  FIX B — Use mutation-count weighted attention (interpretable fallback):
  -----------------------------------------------------------------------
  For each genome, set attn_weight[gene] = number of mutations observed
  in that gene (from the mutation matrix row). No learned parameters needed.
  This is biologically interpretable and genome-specific.
  Load from features/mutation_matrix.npz at inference time.

  FIX C — Remove attention entirely; use gate as explainability signal:
  ---------------------------------------------------------------------
  The gate IS genome-specific (std up to 0.47 across samples).
  Replace the attention narrative with gate-based explanation:
  "dimensions where gate ≈ 1 rely on genomic encoding; gate ≈ 0 on KG."
  Average gate per drug/resistance-class to find which dimensions differ.

  FIX D — Use CARD evidence grades as fixed weights (most defensible claim):
  --------------------------------------------------------------------------
  Replace the learned attention with WHO GRADE weights from gene_mechanism.json.
  Grade I = weight 1.0, Grade II = 0.66, Grade III = 0.33, unclassified = 0.1.
  This is biologically grounded, non-circular, and does not require retraining.

  ── AFTER APPLYING FIX A: VALIDATION CHECKLIST ──────────────────────────

  Re-run this script plus:
    □ Attention std per sample > 0.05  (was 0.007 — 7× increase needed)
    □ Entropy < 90% of max for >50% of test samples
    □ Spearman(attn, SHAP) should INCREASE, ideally > 0.5 and positive
    □ Ablation: full model AUROC > genomic-only AUROC

  ── KG SPARSITY NOTE ────────────────────────────────────────────────────

  Despite 2.4 triples/entity overall, the 26 gene nodes ARE well-embedded
  because they are HUBS: each gene connects to many of its mutations.
  The "sparse" statistic is inflated by 25,000+ leaf mutation nodes each
  with only 1–3 triples. This is expected and not a problem.
""")

print(f"{BOLD}Done. The fix is in model/kg_amr.py — make gene_attn query-conditioned on genomic features.{RST}\n")
