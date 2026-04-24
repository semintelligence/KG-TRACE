# KG-Trace

**Knowledge Graph-Enhanced Antimicrobial Resistance Prediction with Interpretable Cross-Attention**

KG-Trace is a deep learning framework that fuses genomic k-mer features with a knowledge graph (KG) of WHO-curated AMR genes to predict antibiotic resistance in bacterial isolates. A learned cross-attention gate mechanism dynamically balances genomic and KG signals per sample, producing both high-accuracy predictions and gene-level biological explanations.

> **Submission note**: This repository is prepared for double-blind peer review. No author-identifying information is included.

---

## Key Contributions

1. **Cross-attention gate fusion** that adaptively blends genomic and KG embeddings with empirically discriminative behavior between resistant and susceptible samples.
2. **KG pathway traceability**: 94.8% of resistant test samples can be traced to established gene–drug resistance pathways (*katG*, *fabG1*, *inhA*).
3. **Mutation-level SHAP explanations** with significant positive alignment to CARD biological ground truth (Spearman ρ = +0.443, p = 0.023).
4. **Systematic alignment analysis** showing attention and SHAP capture fundamentally different signals — cautioning against interpreting attention as direct biological importance.

---

## Architecture

```
Genomic Features (17,352 k-mers)
        │
        ▼
  Genomic Encoder
  Linear(17352→512) → BN → ReLU → Dropout(0.3)
  Linear(512→256)
        │
        ▼                   KG Encoder
 Genomic Embedding     Cross-Attention over 26
      (256d)         WHO gene embeddings (RotatE 64d)
        │                      │
        └────────┬─────────────┘
                 ▼
    Cross-Attention Gate Fusion (128d)
          α·g_proj + (1−α)·k_proj
                 │
                 ▼
         Resistance Prediction
```

| Component | Details |
|-----------|---------|
| Knowledge Graph | 60,017 triples · 25,095 entities · 6 relation types |
| KG Embeddings | RotatE, 300 training epochs, 64-d complex → magnitude |
| Fusion gate mean | ~0.26 overall; 0.33 (resistant) vs 0.22 (susceptible) |
| Parameters | 9,142,493 |

---

## Results

### Primary Evaluation — *M. tuberculosis* INH (n\_test = 5,665)

| Model | AUROC | F1-macro | Precision | Recall |
|-------|-------|----------|-----------|--------|
| **KG-Trace** | **0.9757** | **0.9594** | **0.9647** | 0.9549 |
| LinearSVC (tuned) | 0.9794 | 0.9585 | 0.9619 | 0.9538 |
| XGBoost (tuned) | 0.9760 | 0.9595 | 0.9628 | 0.9537 |
| Random Forest (tuned) | 0.9806 | 0.9548 | 0.9056 | 0.9130 |

> KG-Trace achieves the highest F1-macro and produces the fewest false negatives (156 vs 168 for the best-tuned baseline), which is the clinically critical error class.

### Multi-Drug Extension — *M. tuberculosis*

| Drug | AUROC | F1-macro |
|------|-------|----------|
| INH  | 0.974 | 0.958 |
| RIF  | 0.985 | 0.965 |
| EMB  | 0.956 | 0.861 |
| LEV  | 0.947 | 0.894 |

### Interpretability Alignment

| Comparison | Spearman ρ | p-value | Interpretation |
|-----------|-----------|---------|----------------|
| SHAP vs CARD | +0.443 | 0.023 | Significant positive — SHAP tracks biology |
| Attention vs CARD | +0.029 | 0.888 | Not significant — attention ≠ importance |
| Attention vs SHAP | −0.438 | 0.025 | Significant inverse — complementary signals |

---

## Repository Structure

```
kg-trace/
├── stage_m1_fasta_qc.py          # Stage 1: FASTA quality control
├── stage_m2_kmer_features.py     # Stage 2: k-mer feature extraction
├── stage_m3_bvbrc_annotations.py # Stage 3: BV-BRC gene annotations
├── stage_m4_labels.py            # Stage 4: resistance label assignment
├── stage_m5_species_kg.py        # Stage 5: species knowledge graph construction
├── stage_m6_train_species.py     # Stage 6: model training
├── stage_m7_explain_species.py   # Stage 7: SHAP + attention explanations
├── stage_m8_results.py           # Stage 8: results compilation
├── paths.py                      # Centralised path configuration
├── requirements.txt              # Python dependencies
├── environment.yml               # Conda environment
├── baselines/                    # Baseline model runs (SVM, XGBoost, RF)
├── data/                         # Data download scripts & logs
├── evaluate/                     # Evaluation scripts & results
├── explain/                      # Figures, SHAP values, alignment metrics
├── features/                     # Feature matrices
├── kg/                           # Knowledge graph files
├── model/                        # Trained model checkpoints & logs
└── paper/                        # Manuscript source (LaTeX)
```

---

## Installation

```bash
# Clone and set up environment
conda env create -f environment.yml
conda activate kg-trace

# Or install via pip
pip install -r requirements.txt
```

**Requirements**: Python ≥ 3.11, PyTorch ≥ 2.1, PyTorch-Lightning ≥ 2.1, PyKEEN ≥ 1.10

---

## Running the Pipeline

Run all stages sequentially from the `kg-trace/` directory:

```bash
python stage_m1_fasta_qc.py          # FASTA quality control
python stage_m2_kmer_features.py     # k-mer feature extraction
python stage_m3_bvbrc_annotations.py # BV-BRC gene annotations
python stage_m4_labels.py            # Resistance label assignment
python stage_m5_species_kg.py        # Build species knowledge graphs
python stage_m6_train_species.py     # Train KG-Trace model
python stage_m7_explain_species.py   # Generate SHAP + attention explanations
python stage_m8_results.py           # Compile and export results
```

To run baselines:

```bash
python baselines/run_baselines.py
```

---

## Data

| Source | Description |
|--------|-------------|
| [PATRIC / BV-BRC](https://bv-brc.org/) | 41,460 bacterial isolate genomes |
| WHO CRyPTIC Consortium | Phenotypic drug susceptibility labels |
| `Mendeley Data/` | Phylogenetic cross-validation fold definitions |

Pre-computed feature matrices and trained checkpoints are provided in the repository. Raw FASTA files are available via the Mendeley Data deposit (see `Zenodo/RELEASE_NOTES.md`).

---

## Key Output Files

| File | Description |
|------|-------------|
| `evaluate/final_results.csv` | Full model comparison table |
| `evaluate/ablation_results.csv` | Architecture ablation results |
| `explain/gene_attention_weights.csv` | Per-genome gene attention weights |
| `explain/fusion_gate_values.csv` | Per-sample cross-attention gate values |
| `explain/alignment_metrics.json` | BCS@10 and Spearman ρ alignment metrics |
| `model/checkpoints/best_model.ckpt` | Best trained model checkpoint |

---

## Reproducibility

All metrics reported in the manuscript are computed directly from model outputs. No values are hardcoded. To reproduce the full evaluation from scratch:

```bash
# Re-run evaluation with existing checkpoint
python evaluate/compile_results.py

# Re-generate all publication figures
python explain/generate_all_figures.py
```

---

## License

Data licensing details are provided in `Zenodo/RELEASE_NOTES.md`. Code is released for research use.
