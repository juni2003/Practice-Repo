"""
DBSCAN: hyperparameters, outliers, and non-convex clustering

This script demonstrates:
- DBSCAN can discover non-convex clusters + label noise as -1
- Sensitivity to eps and min_samples
- Why scaling matters (we scale features before clustering)

Dataset: make_moons + added Gaussian noise points (outliers)

Outputs:
- A grid of scatter plots showing how eps changes results
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def make_data(n: int = 800, noise: float = 0.08, n_outliers: int = 60, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, _ = make_moons(n_samples=n, noise=noise, random_state=seed)

    # Add outliers uniformly in a bounding box
    lo = X.min(axis=0) - 0.7
    hi = X.max(axis=0) + 0.7
    outliers = rng.uniform(lo, hi, size=(n_outliers, 2))

    X_all = np.vstack([X, outliers])
    return X_all


def run_dbscan(X, eps: float, min_samples: int):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels


def plot_grid(X, eps_values, min_samples: int):
    n = len(eps_values)
    cols = 3
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 4.6, rows * 4.2))

    for i, eps in enumerate(eps_values, start=1):
        labels = run_dbscan(X, eps=eps, min_samples=min_samples)
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        plt.subplot(rows, cols, i)
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=18, cmap="tab10", alpha=0.9)
        plt.title(f"eps={eps:.2f}, min_samples={min_samples}\nclusters={n_clusters}, noise={n_noise}")
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("DBSCAN sensitivity to eps (scaled data)", y=1.02, fontsize=14)
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    X = make_data()

    # Scaling is important for distance-based methods.
    Xs = StandardScaler().fit_transform(X)

    eps_values = [0.10, 0.14, 0.18, 0.22, 0.28, 0.35]
    plot_grid(Xs, eps_values=eps_values, min_samples=10)

    print("Notes:")
    print("- Too small eps -> many points become noise, clusters fragment.")
    print("- Too large eps -> clusters merge into one.")
    print("- StandardScaler is usually required unless features already comparable.")


if __name__ == "__main__":
    main()
