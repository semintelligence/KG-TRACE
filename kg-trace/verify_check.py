import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import json

out = np.load('model/test_outputs.npz', allow_pickle=True)
probs = out['probs']
preds = out['preds']
labels = out['labels']
gate = out['gate_values']   # (5665, 128)
attn = out['attn_weights']  # (5665, 26)

# Verify reported metrics
auc = roc_auc_score(labels, probs)
f1 = f1_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
print(f'Verified AUROC: {auc:.4f}')
print(f'Verified F1:    {f1:.4f}')
print(f'Verified Prec:  {prec:.4f}')
print(f'Verified Recall:{rec:.4f}')
print(f'Test n={len(labels)}, n_res={labels.sum()}, n_sens={(labels==0).sum()}')
print()

# Gate analysis - is it saturated?
gate_mean = gate.mean()
frac_zero = (gate == 0).mean()
frac_one  = (gate == 1).mean()
frac_mid  = ((gate > 0.01) & (gate < 0.99)).mean()
print(f'Gate mean (128-dim avg): {gate_mean:.4f}')
print(f'  Gate values == 0.0:     {frac_zero*100:.1f}%')
print(f'  Gate values == 1.0:     {frac_one*100:.1f}%')
print(f'  Gate values [0.01,0.99]:{frac_mid*100:.1f}%')
print(f'  >> Interpretation: if >90% are hard 0 or 1, sigmoid is saturated')
print()

# Attention analysis - is it degenerate (uniform)?
max_ent = np.log(26)
attn_ent = -(attn * np.log(attn + 1e-12)).sum(axis=1)  # per sample
frac_unif = (attn_ent > 0.99 * max_ent).mean()
frac_half = (attn_ent > 0.9 * max_ent).mean()
print(f'Max possible attention entropy (uniform over 26): {max_ent:.4f}')
print(f'Mean attention entropy over test set:             {attn_ent.mean():.4f}')
print(f'Fraction samples with entropy > 99% of max:      {frac_unif*100:.1f}%')
print(f'Fraction samples with entropy > 90% of max:      {frac_half*100:.1f}%')
# Std of attention weights per sample
attn_std = attn.std(axis=1)
print(f'Mean std of attention per sample:                 {attn_std.mean():.6f}')
print(f'  (uniform over 26 genes would have std = {(1/26*(1 - 1/26))**0.5:.6f})')
print()

# Spearman correlation check
from scipy import stats
mean_attn = attn.mean(axis=0)  # (26,)
import pandas as pd
shap_df = pd.read_csv('explain/shap_values.csv', index_col=0)
print('SHAP CSV cols:', shap_df.columns[:5].tolist(), '...', shap_df.shape)
