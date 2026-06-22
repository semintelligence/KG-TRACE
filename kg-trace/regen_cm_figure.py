"""Regenerate confusion matrix from test_outputs.npz (v1 checkpoint)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os

d = np.load('/Users/namangarg/Desktop/KG-Trace/kg-trace/model/test_outputs.npz', allow_pickle=True)
labels = d['labels']
preds = d['preds']
probs = d['probs']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Figure 3: Test Set Performance (n={len(labels):,})', fontsize=16, fontweight='bold', y=1.02)

# A) ROC Curve (KG-TRACE only from v1 checkpoint)
ax = axes[0]
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)
ax.plot(fpr, tpr, color='#E74C3C', lw=2, label=f'KG-TRACE (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('A) ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.01])

# B) Confusion Matrix
ax = axes[1]
cm = confusion_matrix(labels, preds)
# cm[0,0]=TN, cm[0,1]=FP, cm[1,0]=FN, cm[1,1]=TP
total = cm.sum()
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('B) KG-TRACE Confusion Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        pct = val / total * 100
        color = 'white' if val > total/4 else 'black'
        ax.text(j, i, f'{val:,}\n({pct:.1f}%)', ha='center', va='center',
                fontsize=14, color=color, fontweight='bold')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Susceptible', 'Resistant'], fontsize=11)
ax.set_yticklabels(['Susceptible', 'Resistant'], fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)

plt.tight_layout()

# Save
out_path = '/Users/namangarg/Desktop/KG-Trace/kg-trace/paper/KG_TRACE_Confusion_Matrix.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f'Saved to {out_path}')
print(f'AUROC: {roc_auc:.4f}')
print(f'CM: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')
