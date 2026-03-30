"""
Step 7: Run Baselines — SVM, XGBoost, Random Forest
Uses IDENTICAL train/test splits as KG-AMR v2.
All metrics computed from real predictions, never hardcoded.
"""
import sys, os, time, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import PROJECT_DIR

cwd = os.getcwd()
assert "kg-amr-v2" in cwd or "AMR" in cwd, f"ABORT: wrong dir {cwd}"

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = os.path.join(PROJECT_DIR, "features")
MODEL_DIR = os.path.join(PROJECT_DIR, "model")
BASELINES_DIR = os.path.join(PROJECT_DIR, "baselines")
os.makedirs(BASELINES_DIR, exist_ok=True)

DRUG = "INH"

# ── 1. Load same data and splits as KG-AMR v2 ───────────────────────────────
print("[1/4] Loading data with IDENTICAL splits as KG-AMR v2...")
t0 = time.time()

# Load mutation matrix
X_sparse = sparse.load_npz(os.path.join(FEATURES_DIR, "mutation_matrix.npz"))
with open(os.path.join(FEATURES_DIR, "mutation_samples.json")) as f:
    all_samples = json.load(f)
with open(os.path.join(FEATURES_DIR, "mutation_features.json")) as f:
    all_features = json.load(f)

# Load labels
labels_df = pd.read_parquet(os.path.join(FEATURES_DIR, f"labels/labels_{DRUG}.parquet"))
labels_df = labels_df[~labels_df.index.duplicated(keep="first")]

# Load split IDs from training
with open(os.path.join(MODEL_DIR, "split_ids.json")) as f:
    split_info = json.load(f)

train_ids = set(split_info["train_ids"])
val_ids = set(split_info["val_ids"])
test_ids = set(split_info["test_ids"])

# Recreate sample_to_idx
sample_to_idx = {s: i for i, s in enumerate(all_samples)}

# Build train (train + val for baselines — they don't need val), test
train_val_samples = [s for s in all_samples if s in train_ids or s in val_ids]
test_samples = [s for s in all_samples if s in test_ids]

train_val_indices = [sample_to_idx[s] for s in train_val_samples]
test_indices = [sample_to_idx[s] for s in test_samples]

X_train = X_sparse[train_val_indices].toarray().astype(np.float32)
X_test = X_sparse[test_indices].toarray().astype(np.float32)
y_train = labels_df.loc[train_val_samples, "label"].values.astype(int)
y_test = labels_df.loc[test_samples, "label"].values.astype(int)

print(f"  Train+Val: {X_train.shape[0]} ({100*(y_train==1).mean():.1f}% R)")
print(f"  Test:      {X_test.shape[0]} ({100*(y_test==1).mean():.1f}% R)")
print(f"  Features:  {X_train.shape[1]}")

# ── 2. Define and run baselines ──────────────────────────────────────────────
print("\n[2/4] Training baselines...")

results = []

# --- SVM ---
print("\n  --- SVM (LinearSVC + Platt calibration) ---")
t1 = time.time()
svm_base = LinearSVC(max_iter=5000, random_state=42, dual="auto")
svm_cal = CalibratedClassifierCV(svm_base, cv=3)
svm_cal.fit(X_train, y_train)
svm_preds = svm_cal.predict(X_test)
svm_probs = svm_cal.predict_proba(X_test)[:, 1]
svm_time = time.time() - t1

svm_metrics = {
    "model": "SVM",
    "auroc": float(roc_auc_score(y_test, svm_probs)),
    "f1_macro": float(f1_score(y_test, svm_preds, average="macro")),
    "precision_macro": float(precision_score(y_test, svm_preds, average="macro")),
    "recall_macro": float(recall_score(y_test, svm_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test, svm_preds).tolist(),
    "train_time_s": round(svm_time, 1),
}
results.append(svm_metrics)
print(f"    AUROC={svm_metrics['auroc']:.4f}, F1-macro={svm_metrics['f1_macro']:.4f} ({svm_time:.1f}s)")
print(f"    CM: {svm_metrics['confusion_matrix']}")

# --- XGBoost ---
print("\n  --- XGBoost ---")
t1 = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    random_state=42, eval_metric="logloss",
    tree_method="hist",  # CPU-friendly
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_time = time.time() - t1

xgb_metrics = {
    "model": "XGBoost",
    "auroc": float(roc_auc_score(y_test, xgb_probs)),
    "f1_macro": float(f1_score(y_test, xgb_preds, average="macro")),
    "precision_macro": float(precision_score(y_test, xgb_preds, average="macro")),
    "recall_macro": float(recall_score(y_test, xgb_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist(),
    "train_time_s": round(xgb_time, 1),
}
results.append(xgb_metrics)
print(f"    AUROC={xgb_metrics['auroc']:.4f}, F1-macro={xgb_metrics['f1_macro']:.4f} ({xgb_time:.1f}s)")
print(f"    CM: {xgb_metrics['confusion_matrix']}")

# --- Random Forest --- (100 trees, sqrt features to stay within memory)
print("\n  --- Random Forest ---")
t1 = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=20, max_features="sqrt",
    random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_time = time.time() - t1

rf_metrics = {
    "model": "RandomForest",
    "auroc": float(roc_auc_score(y_test, rf_probs)),
    "f1_macro": float(f1_score(y_test, rf_preds, average="macro")),
    "precision_macro": float(precision_score(y_test, rf_preds, average="macro")),
    "recall_macro": float(recall_score(y_test, rf_preds, average="macro")),
    "confusion_matrix": confusion_matrix(y_test, rf_preds).tolist(),
    "train_time_s": round(rf_time, 1),
}
results.append(rf_metrics)
print(f"    AUROC={rf_metrics['auroc']:.4f}, F1-macro={rf_metrics['f1_macro']:.4f} ({rf_time:.1f}s)")
print(f"    CM: {rf_metrics['confusion_matrix']}")

# ── 3. SHAP analysis for baselines ──────────────────────────────────────────
print("\n[3/4] Running SHAP on baselines...")
import shap, gc

# Free training matrix before SHAP to recover memory
del X_train
gc.collect()

# Smaller subsets: 300 test samples, 50 background
rng = np.random.RandomState(42)
shap_test_size = min(300, X_test.shape[0])
shap_test_idx = rng.choice(X_test.shape[0], shap_test_size, replace=False)
X_shap_test = X_test[shap_test_idx]
bg_size = 50
bg_idx = rng.choice(X_shap_test.shape[0], bg_size, replace=False)
X_bg = X_shap_test[bg_idx]
print(f"  SHAP test subset: {shap_test_size}, background: {bg_size}")

# --- SHAP for SVM (use coefficients directly) ---
print("  SHAP: SVM (coefficient-based importance)...")
try:
    # LinearSVC: use model coefficients as feature importance (equivalent to linear SHAP)
    svm_coefs = np.abs(svm_base.coef_[0])
    svm_top_idx = np.argsort(svm_coefs)[::-1][:20]
    svm_top_features = [(all_features[i], float(svm_coefs[i])) for i in svm_top_idx]
    svm_shap_values = None  # not needed for BCS
    print(f"    Top-5 SVM: {[f[0] for f in svm_top_features[:5]]}")
except Exception as e:
    print(f"    SVM coef extraction failed: {e}")
    svm_shap_values = None
    svm_top_features = []

# --- SHAP for XGBoost (TreeExplainer) ---
print("  SHAP: XGBoost (TreeExplainer)...")
try:
    xgb_explainer = shap.TreeExplainer(xgb_model)
    xgb_shap_values = xgb_explainer.shap_values(X_shap_test)
    xgb_shap_mean = np.abs(xgb_shap_values).mean(axis=0)
    xgb_top_idx = np.argsort(xgb_shap_mean)[::-1][:20]
    xgb_top_features = [(all_features[i], float(xgb_shap_mean[i])) for i in xgb_top_idx]
    print(f"    Top-5 XGB SHAP: {[f[0] for f in xgb_top_features[:5]]}")
except Exception as e:
    print(f"    XGB SHAP failed: {e}")
    xgb_shap_values = None
    xgb_top_features = []

# --- Random Forest: use feature_importances_ (Gini) for speed, then SHAP on small subset ---
print("  SHAP: Random Forest (feature_importances_ + TreeExplainer on subset)...")
rf_shap_values = None
try:
    # Use built-in Gini importance as primary ranking
    rf_imp = rf_model.feature_importances_
    rf_top_idx_gini = np.argsort(rf_imp)[::-1][:20]
    rf_top_features = [(all_features[i], float(rf_imp[i])) for i in rf_top_idx_gini]
    print(f"    Top-5 RF (Gini): {[f[0] for f in rf_top_features[:5]]}")
    # Also run TreeExplainer on small subset for SHAP values
    rf_shap_subset = min(200, shap_test_size)
    rf_explainer = shap.TreeExplainer(rf_model)
    rf_shap_values = rf_explainer.shap_values(X_shap_test[:rf_shap_subset])
    if isinstance(rf_shap_values, list):
        rf_shap_values = rf_shap_values[1]
    print(f"    RF SHAP computed on {rf_shap_subset} samples")
except Exception as e:
    print(f"    RF SHAP failed: {e}")
    rf_top_features = []

# ── 4. Compute BCS for each baseline ────────────────────────────────────────
print("\n[4/4] Computing BCS for baselines...")

# Load CARD gene set (the 26 catalogue genes)
with open(os.path.join(PROJECT_DIR, "kg/gene_mechanism.json")) as f:
    gene_mechanism = json.load(f)
card_gene_set = set(gene_mechanism.keys())

def compute_bcs(top_features_list, card_gene_set, N=10):
    """BCS = fraction of top-N SHAP features whose gene maps to CARD catalogue."""
    gene_names = []
    for feat_name, _ in top_features_list[:N]:
        gene = feat_name.split(":")[0]
        gene_names.append(gene)
    mapped = [g for g in gene_names if g in card_gene_set]
    bcs = len(mapped) / N if N > 0 else 0.0
    return bcs, gene_names, mapped

for model_name, top_feats, metrics in [
    ("SVM", svm_top_features, svm_metrics),
    ("XGBoost", xgb_top_features, xgb_metrics),
    ("RandomForest", rf_top_features, rf_metrics),
]:
    if top_feats:
        bcs, genes, mapped = compute_bcs(top_feats, card_gene_set, N=10)
        metrics["bcs_10"] = float(bcs)
        metrics["top10_genes"] = genes
        metrics["top10_mapped"] = mapped
        print(f"  {model_name}: BCS@10 = {bcs:.2f} ({len(mapped)}/10 mapped)")
        print(f"    Top-10 genes: {genes}")
        print(f"    Mapped to CARD: {mapped}")
    else:
        metrics["bcs_10"] = None
        print(f"  {model_name}: BCS not computed (SHAP failed)")

# ── Save results ─────────────────────────────────────────────────────────────
# Save baseline results CSV
csv_path = os.path.join(BASELINES_DIR, "baseline_results.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model", "auroc", "f1_macro", "precision_macro", "recall_macro",
        "confusion_matrix", "bcs_10", "train_time_s"
    ])
    writer.writeheader()
    for r in results:
        row = {k: r.get(k, "") for k in writer.fieldnames}
        row["confusion_matrix"] = str(r.get("confusion_matrix", ""))
        writer.writerow(row)

# Save full results JSON
with open(os.path.join(BASELINES_DIR, "baseline_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save SHAP top features
shap_data = {
    "SVM": svm_top_features,
    "XGBoost": xgb_top_features,
    "RandomForest": rf_top_features,
}
with open(os.path.join(BASELINES_DIR, "baseline_shap_top_features.json"), "w") as f:
    json.dump(shap_data, f, indent=2)

# Save raw SHAP values for baselines
shap_arrays = {}
if svm_shap_values is not None:
    shap_arrays["svm_shap"] = svm_shap_values
if xgb_shap_values is not None:
    shap_arrays["xgb_shap"] = xgb_shap_values
if rf_shap_values is not None:
    shap_arrays["rf_shap"] = rf_shap_values
if shap_arrays:
    np.savez_compressed(os.path.join(BASELINES_DIR, "baseline_shap_values.npz"), **shap_arrays)

# Save test predictions for confusion matrices in dashboard
np.savez(
    os.path.join(BASELINES_DIR, "baseline_predictions.npz"),
    svm_preds=svm_preds, svm_probs=svm_probs,
    xgb_preds=xgb_preds, xgb_probs=xgb_probs,
    rf_preds=rf_preds, rf_probs=rf_probs,
    y_test=y_test, test_ids=np.array(test_samples),
)

elapsed = time.time() - t0
print(f"\n  Total elapsed: {elapsed:.1f}s")
print(f"  Results saved to: {BASELINES_DIR}")

# Print comparison table
print("\n  === Baseline Comparison ===")
print(f"  {'Model':<15s} {'AUROC':>8s} {'F1-macro':>10s} {'BCS@10':>8s}")
print(f"  {'-'*43}")
for r in results:
    bcs_str = f"{r['bcs_10']:.2f}" if r.get('bcs_10') is not None else "n/a"
    print(f"  {r['model']:<15s} {r['auroc']:>8.4f} {r['f1_macro']:>10.4f} {bcs_str:>8s}")

print("\nDONE")
