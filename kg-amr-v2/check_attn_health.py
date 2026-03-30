import numpy as np
from collections import Counter

out  = np.load('model/test_outputs.npz', allow_pickle=True)
attn = out['attn_weights']   # [n_test, 26]
H     = -(attn * np.log(attn + 1e-12)).sum(1)
H_max = np.log(attn.shape[1])

print(f'Entropy:        {H.mean():.4f} / {H_max:.4f}  (target: <2.50)')
print(f'Near-uniform:   {(H > 0.99*H_max).mean()*100:.1f}%          (target: <20%)')
print(f'Attn std:       {attn.std(1).mean():.4f}               (target: >0.05)')

gate = out['gate_values']
print(f'Gate active:    {((gate>0.01)&(gate<0.99)).mean()*100:.1f}%          (target: >40%)')

# Was improvement over baseline?
print(f'\nBefore fix: Entropy=3.2386, Near-uniform=100%, Attn std=0.007, Gate active=11.8%')

# Top-gene distribution
gene_names = out['gene_names']
top_genes  = attn.argmax(axis=1)
top_counts = Counter(gene_names[i] for i in top_genes)
print('\nTop attended gene distribution (most common 10):')
for gene, cnt in top_counts.most_common(10):
    bar = chr(9608) * int(cnt / len(top_genes) * 100)
    print(f'  {gene:<10} {cnt:>5} ({cnt/len(top_genes)*100:.1f}%) {bar}')

# How many unique top-genes across the test set?
n_unique = len(top_counts)
print(f'\nUnique top genes across test set: {n_unique}/26')
print(f'(before fix: all 5665 samples had the same top gene)')
