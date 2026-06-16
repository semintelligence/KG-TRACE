import os
import numpy as np
from sklearn.metrics import brier_score_loss

def calculate_ece(probs, labels, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            bin_probs = probs[mask]
            bin_labels = labels[mask]
            
            avg_pred = np.mean(bin_probs)
            avg_true = np.mean(bin_labels)
            
            ece += len(bin_probs) * np.abs(avg_pred - avg_true)
            
    return ece / len(probs)

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "..", "model", "test_outputs.npz")
    data = np.load(file_path)
    # The probs array might be 2D depending on the output (e.g. (N, 2)). We need P(y=1)
    probs = data["probs"]
    if len(probs.shape) == 2 and probs.shape[1] == 2:
        probs = probs[:, 1]
    labels = data["labels"]

    brier = brier_score_loss(labels, probs)
    ece = calculate_ece(probs, labels)

    print("=== Calibration Metrics ===")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
