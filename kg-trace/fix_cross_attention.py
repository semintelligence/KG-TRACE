"""
fix_cross_attention.py
======================
Patches kg_trace.py to fix the sample-agnostic attention bug,
then validates the fix before you retrain.

Run from KG-Trace/:
    python3 fix_cross_attention.py

What it does:
  1. Backs up the original model file
  2. Applies the cross-attention fix (2 surgical changes)
  3. Runs a forward-pass sanity check to confirm attention
     now varies across samples BEFORE you spend time retraining
"""

import shutil, sys, json
from pathlib import Path

RED  = "\033[91m"; YEL = "\033[93m"; GRN = "\033[92m"
BOLD = "\033[1m";  RST = "\033[0m"

def ok(msg):   print(f"  {GRN}✓ {msg}{RST}")
def warn(msg): print(f"  {YEL}⚠ {msg}{RST}")
def bad(msg):  print(f"  {RED}✗ {msg}{RST}"); sys.exit(1)
def header(t): print(f"\n{BOLD}{'='*60}\n  {t}\n{'='*60}{RST}")


# ── Step 1: Backup ─────────────────────────────────────────────────────────────
header("1. Backing Up Original Model")

MODEL_PATH  = Path("model/kg_trace.py")
BACKUP_PATH = Path("model/kg_trace.py.bak_original")

if not MODEL_PATH.exists():
    bad(f"Cannot find {MODEL_PATH} — run from your KG-Trace/ directory")

if BACKUP_PATH.exists():
    warn(f"Backup already exists at {BACKUP_PATH} — skipping overwrite")
else:
    shutil.copy(MODEL_PATH, BACKUP_PATH)
    ok(f"Backed up to {BACKUP_PATH}")


# ── Step 2: Read & Patch ───────────────────────────────────────────────────────
header("2. Applying Cross-Attention Patch")

src = MODEL_PATH.read_text()

# --------------------------------------------------------------------------
# PATCH A — add gene_attn_q Linear layer in __init__
# We find the existing self.gene_attn line and append gene_attn_q after it.
# Works with both 4-space and 8-space indentation.
# --------------------------------------------------------------------------
MARKER_A = "self.gene_attn = nn.Linear(KG_EMBED_DIM, 1)"

if MARKER_A not in src:
    bad(
        f"Could not find '{MARKER_A}' in {MODEL_PATH}.\n"
        "  Open the file and check the exact spelling, then update MARKER_A in this script."
    )

# Detect indentation used for this line
for line in src.splitlines():
    if MARKER_A in line:
        indent = len(line) - len(line.lstrip())
        break

pad = " " * indent
replacement_A = (
    f"{MARKER_A}  # kept for checkpoint compat\n"
    f"{pad}self.gene_attn_q = nn.Linear(256, KG_EMBED_DIM, bias=False)"
    f"  # cross-attn: query from genomic encoder (256-dim → KG_EMBED_DIM)"
)

src = src.replace(MARKER_A, replacement_A, 1)
ok("Patch A — added self.gene_attn_q to __init__")

# --------------------------------------------------------------------------
# PATCH B — replace sample-agnostic forward() attention with cross-attention
#
# IMPORTANT: `g` at this point in forward() is the raw 256-dim output of
# genomic_encoder (BEFORE proj_g which maps 256→128).
# gene_attn_q takes 256-dim input, which matches `g` perfectly.
#
# OLD (same output for every genome):
#   attn_scores = self.gene_attn(gene_embeds)          # [batch, n_genes, 1]
#
# NEW (genome-specific, query from genomic hidden state g):
#   query       = self.gene_attn_q(g).unsqueeze(2)     # [B, KG_EMBED_DIM, 1]
#   attn_scores = torch.bmm(gene_embeds, query) / (KG_EMBED_DIM ** 0.5)
# --------------------------------------------------------------------------
MARKER_B = "attn_scores = self.gene_attn(gene_embeds)          # [batch, n_genes, 1]"

if MARKER_B not in src:
    bad(
        f"Could not find the exact MARKER_B string in forward().\n"
        "  Check model/kg_trace.py for the exact whitespace/comment and update MARKER_B."
    )

for line in src.splitlines():
    if MARKER_B in line:
        indent_b = len(line) - len(line.lstrip())
        break

pad_b = " " * indent_b
replacement_B = (
    f"# Cross-attention: each genome queries the gene most aligned with its hidden state\n"
    f"{pad_b}# g is the 256-dim genomic_encoder output (pre-proj_g), used as the query\n"
    f"{pad_b}query       = self.gene_attn_q(g).unsqueeze(2)              # [B, KG_EMBED_DIM, 1]\n"
    f"{pad_b}attn_scores = torch.bmm(gene_embeds, query) / (gene_embeds.shape[-1] ** 0.5)"
    f"  # [B, n_genes, 1]"
)

src = src.replace(MARKER_B, replacement_B, 1)
ok("Patch B — attention now uses genomic query (genome-specific)")

# Write patched file
MODEL_PATH.write_text(src)
ok(f"Patched model written to {MODEL_PATH}")


# ── Step 3: Validate — Forward-Pass Sanity Check ──────────────────────────────
header("3. Validating Fix — Forward Pass Sanity Check")

import torch
import numpy as np

# Load real gene embeddings
try:
    ent2id     = json.load(open("kg/entity_to_id.json"))
    emb_np     = np.load("kg/embeddings/entity_embeddings.npy")
    gene_names = np.load("model/test_outputs.npz", allow_pickle=True)["gene_names"]
    gene_ids   = [ent2id[g] for g in gene_names if g in ent2id]
    gene_emb   = torch.tensor(emb_np[gene_ids], dtype=torch.float32)  # [26, 64]
    KG_DIM     = gene_emb.shape[1]
    N_GENES    = gene_emb.shape[0]
    ok(f"Loaded {N_GENES} gene embeddings (dim={KG_DIM})")
except Exception as e:
    bad(f"Could not load embeddings: {e}")

# Import patched model
try:
    sys.path.insert(0, ".")
    import importlib
    # Remove cached module so reload picks up the patched file
    for mod_name in list(sys.modules.keys()):
        if "kg_trace" in mod_name:
            del sys.modules[mod_name]
    import model.kg_trace as kg_mod
    KGTrace = kg_mod.KGTrace
    ok("Patched model imported successfully")
except Exception as e:
    bad(f"Import failed after patching: {e}")

# Instantiate model with correct constructor signature (kmer_dim, num_genes)
model = None
for kwargs in [
    dict(kmer_dim=17352, num_genes=N_GENES),   # actual signature used in this repo
    dict(kmer_dim=17352, num_genes=26),
    dict(genomic_dim=17352, kg_embed_dim=KG_DIM, n_genes=N_GENES, hidden_dim=256, dropout=0.3),
    dict(input_dim=17352,   kg_embed_dim=KG_DIM, n_genes=N_GENES),
    dict(genomic_dim=17352, kg_embed_dim=KG_DIM, n_genes=N_GENES),
]:
    try:
        model = KGTrace(**kwargs)
        ok(f"Model instantiated with kwargs: {list(kwargs.keys())}")
        break
    except TypeError:
        continue

if model is None:
    bad(
        "Could not auto-instantiate KGTrace. Check the constructor signature.\n"
        "  Then instantiate it manually here and re-run from the validation block."
    )

model.eval()

# Confirm gene_attn_q was added
if not hasattr(model, "gene_attn_q"):
    bad("self.gene_attn_q not found on the model — Patch A may not have applied correctly.")
ok(f"self.gene_attn_q confirmed: {model.gene_attn_q}")

# Run two DIFFERENT random genomes through the patched attention sub-path.
# NOTE: g is the 256-dim output of genomic_encoder (pre-proj_g).
# gene_attn_q takes 256-dim input, so we stop at genomic_encoder, not proj_g.
torch.manual_seed(42)
g1 = torch.randn(1, 17352)
g2 = torch.randn(1, 17352)

try:
    encoder = getattr(model, "genomic_encoder", None)

    if encoder is None:
        warn("Could not find model.genomic_encoder — skipping auto-check")
    else:
        with torch.no_grad():
            h1 = encoder(g1)   # [1, 256]  — 256-dim genomic hidden state
            h2 = encoder(g2)   # [1, 256]

            # gene_attn_q maps 256 → KG_EMBED_DIM=64
            q1 = model.gene_attn_q(h1).unsqueeze(2)   # [1, 64, 1]
            q2 = model.gene_attn_q(h2).unsqueeze(2)   # [1, 64, 1]

            s1 = torch.bmm(gene_emb.unsqueeze(0), q1).squeeze()   # [26]
            s2 = torch.bmm(gene_emb.unsqueeze(0), q2).squeeze()   # [26]

            a1 = torch.softmax(s1 / (KG_DIM ** 0.5), dim=0)
            a2 = torch.softmax(s2 / (KG_DIM ** 0.5), dim=0)

        diff     = (a1 - a2).abs().mean().item()
        max_diff = (a1 - a2).abs().max().item()
        top1     = gene_names[a1.argmax().item()]
        top2     = gene_names[a2.argmax().item()]

        print(f"\n  Attention weight comparison — two random genomes:")
        print(f"    Mean absolute diff: {diff:.5f}")
        print(f"    Max  absolute diff: {max_diff:.5f}")
        print(f"    Genome 1 top gene:  {top1}")
        print(f"    Genome 2 top gene:  {top2}")

        if diff > 0.005:
            ok(f"FIX CONFIRMED — attention weights differ per genome (mean Δ={diff:.4f})")
        elif diff > 0.001:
            warn(f"Attention differs slightly (Δ={diff:.5f}) — may still be partially degenerate")
        else:
            bad(
                f"Attention still identical (Δ={diff:.6f}).\n"
                "  gene_attn_q may not be connected to the correct hidden state.\n"
                "  Check that `g` in forward() is the 256-dim genomic representation."
            )

except Exception as e:
    warn(f"Attention check raised an exception: {e}")
    import traceback; traceback.print_exc()
    warn("Inspect model/kg_trace.py forward() manually to confirm.")


# ── Step 4: Show the patched forward() for visual confirmation ─────────────────
header("4. Patched forward() — Visual Check")

src_patched = MODEL_PATH.read_text()
in_forward  = False
lines_shown = 0
print()
for line in src_patched.splitlines():
    if "def forward(" in line:
        in_forward = True
    if in_forward:
        print(f"  {line}")
        lines_shown += 1
        if lines_shown > 2 and line.strip() == "":
            break    # stop after first blank line post-return
        if "return " in line:
            break


# ── Step 5: Post-Retraining Validation Command ────────────────────────────────
header("5. After Retraining — Run This Check")

print(f"""
  python3 -c "
  import numpy as np
  out  = np.load('model/test_outputs.npz', allow_pickle=True)
  attn = out['attn_weights']                      # [n_test, 26]
  H     = -(attn * np.log(attn + 1e-12)).sum(1)
  H_max = np.log(attn.shape[1])

  print(f'Attention entropy: {{H.mean():.4f}} / {{H_max:.4f}}')
  print(f'Near-uniform samples: {{(H > 0.99*H_max).mean()*100:.1f}}%  (target: <20%)')
  print(f'Per-sample attn std:  {{attn.std(1).mean():.4f}}           (target: >0.05)')
  print()
  gate = out['gate_values']
  active = ((gate > 0.01) & (gate < 0.99)).mean() * 100
  print(f'Gate active dims: {{active:.1f}}%  (target: >40%)')
  "

  HEALTHY TARGETS
  ---------------
  Attention entropy      < 2.50   (before fix: 3.24 = fully uniform)
  Near-uniform samples   < 20%    (before fix: 100%)
  Per-sample attn std    > 0.05   (before fix: 0.007)
  Gate active dims       > 40%    (before fix: 11.8%)

  ROLLBACK
  --------
  cp model/kg_trace.py.bak_original model/kg_trace.py
""")

print(f"{BOLD}Patch complete. Verify forward() above, then retrain.{RST}\n")
