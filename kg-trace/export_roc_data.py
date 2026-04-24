#!/usr/bin/env python3
"""Export ground truth labels, predicted probabilities, and predicted labels to CSV."""
import numpy as np
import csv, os

# ── Load data ──
to = np.load("model/test_outputs.npz", allow_pickle=True)
bp = np.load("baselines/baseline_predictions.npz", allow_pickle=True)

# ── Print summary to console ──
print("=" * 60)
print("KG-TRACE TEST OUTPUTS")
print(f"  Total test samples : {len(to['labels'])}")
print(f"  Resistant (1)      : {int(to['labels'].sum())}")
print(f"  Susceptible (0)    : {int((to['labels']==0).sum())}")
print(f"  Probs range        : [{to['probs'].min():.6f}, {to['probs'].max():.6f}]")
print()
print("BASELINE PREDICTIONS")
print(f"  Total test samples : {len(bp['y_test'])}")
print(f"  SVM probs range    : [{bp['svm_probs'].min():.6f}, {bp['svm_probs'].max():.6f}]")
print(f"  RF  probs range    : [{bp['rf_probs'].min():.6f}, {bp['rf_probs'].max():.6f}]")
print("=" * 60)

# ── 1) Export KG-TRACE: labels + probs + preds ──
out1 = "explain/final_publication_figures/kgtrace_predictions.csv"
with open(out1, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_index", "ground_truth_label", "kgtrace_probability", "kgtrace_predicted_label"])
    for i in range(len(to["labels"])):
        w.writerow([i, int(to["labels"][i]), round(float(to["probs"][i]), 8), int(to["preds"][i])])
print(f"\n✓ Saved KG-TRACE predictions → {out1}  ({len(to['labels'])} rows)")

# ── 2) Export Baselines: labels + svm_probs + rf_probs ──
out2 = "explain/final_publication_figures/baseline_predictions.csv"
with open(out2, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_index", "ground_truth_label", "svm_probability", "rf_probability"])
    for i in range(len(bp["y_test"])):
        w.writerow([i, int(bp["y_test"][i]), round(float(bp["svm_probs"][i]), 8), round(float(bp["rf_probs"][i]), 8)])
print(f"✓ Saved Baseline predictions → {out2}  ({len(bp['y_test'])} rows)")

# ── 3) Print first 30 rows to console for quick inspection ──
print("\n" + "=" * 80)
print("FIRST 30 ROWS: KG-TRACE")
print(f"{'Index':>6}  {'Truth':>6}  {'Prob':>12}  {'Pred':>6}")
print("-" * 36)
for i in range(min(30, len(to["labels"]))):
    print(f"{i:>6}  {int(to['labels'][i]):>6}  {float(to['probs'][i]):>12.8f}  {int(to['preds'][i]):>6}")

print("\n" + "=" * 80)
print("FIRST 30 ROWS: BASELINES")
print(f"{'Index':>6}  {'Truth':>6}  {'SVM Prob':>12}  {'RF Prob':>12}")
print("-" * 48)
for i in range(min(30, len(bp["y_test"]))):
    print(f"{i:>6}  {int(bp['y_test'][i]):>6}  {float(bp['svm_probs'][i]):>12.8f}  {float(bp['rf_probs'][i]):>12.8f}")

print("\nDone! CSVs are ready for ROC curve and confusion matrix recreation.")
