# KG-Trace — Session Notes

## Project Summary
KG-Trace is a hybrid deep-learning model for antimicrobial resistance (AMR) prediction that fuses binary mutation features with knowledge graph (KG) embeddings via cross-attention gating.

## Architecture
- **Genomic encoder**: Linear(17352→512)→BN→ReLU→Drop(0.3)→Linear(512→256)
- **KG encoder**: Self-attention pooling over 26 WHO catalogue gene embeddings (RotatE, 64d)
- **Fusion**: Cross-attention gate MLP: σ(MLP([g_proj;k_proj])) → gate·g + (1-gate)·k (128d)
- **Heads**: AMR classifier (128→64→2), Gene detection (128→64→26→Sigmoid)
- **Training**: PyTorch Lightning, Adam lr=1e-3, 100 epochs, early stopping patience=10, MPS backend

## Key Results

### INH Baseline (100 epochs, n_test=5,665)
| Model | AUROC | F1-macro |
|-------|-------|----------|
| KG-Trace | 0.9740 | 0.9580 |
| SVM | 0.9787 | 0.9576 |
| XGBoost | 0.9764 | 0.9579 |
| Random Forest | 0.9604 | 0.8350 |

### Multi-Drug Extension (10 epochs each)
| Drug | AUROC | F1-macro | N_test |
|------|-------|----------|--------|
| INH | 0.9740 | 0.9580 | 5,665 |
| RIF | 0.9850 | 0.9647 | 5,675 |
| EMB | 0.9561 | 0.8610 | 5,324 |
| LEV | 0.9473 | 0.8938 | 2,267 |

### Ablation Study (10 epochs each)
| Config | AUROC | Delta vs Full |
|--------|-------|---------------|
| genomic_only | 0.9780 | +0.0058 |
| kg_only | 0.9473 | −0.0249 |
| avg_pool_no_attention | 0.9748 | +0.0026 |
| scalar_fusion | 0.9773 | +0.0051 |
| full_kg_trace | 0.9722 | baseline |

## Honest Observations
1. **SVM baseline outperforms KG-Trace** on AUROC (0.9787 vs 0.9740). KG-Trace's value is interpretability, not raw classification.
2. **Genomic-only ablation (0.9780) > full model (0.9722)** at 10 epochs — KG adds interpretability more than discriminative power.
3. **Mendeley non-MTB datasets**: Only labels available locally — no genome sequences for feature extraction. KG is MTB-specific.
4. All metrics computed from actual model outputs, zero hardcoded values.

## Data Pipeline
- **Input**: CRyPTIC/Zenodo M. tuberculosis WGS → binary mutation matrix (41,460 samples × 17,352 features)
- **KG**: 60,017 triples, 25,095 entities, 6 relations (from CARD + resistance literature)
- **Embeddings**: RotatE (PyKEEN), 300 epochs → 25,095 × 64 complex → magnitude
- **Split**: Stratified 70:15:15 (train:val:test), saved in split_ids.json

## Generated Artifacts
- 10 publication figures in `explain/figures/` (.html + .png + .pdf)
- Interactive dashboard: `explain/report.html`
- 5,666 per-genome JSON reports in `explain/per_genome_reports/`
- All result CSVs: final_results.csv, ablation_results.csv, multi_dataset_results.csv
- Model checkpoint: model/checkpoints/best_model.ckpt (109 MB)

## File Inventory
See final file listing produced at end of session.
