"""
HDBSCAN (optional dependency)

HDBSCAN is often more robust than DBSCAN because it can handle:
- variable density clusters
- automatic selection of stable clusters from a hierarchy

This script:
- generates blobs with different densities
- runs HDBSCAN
- plots clustering results and basic stats

Install:
  pip install hdbscan

If hdbscan isn't installed, the script exits with instructions.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
except ImportError:
    raise SystemExit("hdbscan not installed. Run: pip install hdbscan")


def make_var_density_blobs(seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = [(-3, -2), (0, 0), (3, 2)]
    cluster_std = [0.3, 0.9, 0.45]  # different densities
    X, _ = make_blobs(n_samples=[350, 550, 400], centers=centers, cluster_std=cluster_std, random_state=seed)
    # add mild uniform noise
    noise = rng.uniform(low=-6, high=6, size=(80, 2))
    return np.vstack([X, noise])


def main() -> None:
    X = make_var_density_blobs()
    Xs = StandardScaler().fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10)
    labels = clusterer.fit_predict(Xs)

    n_noise = int(np.sum(labels == -1))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"HDBSCAN clusters: {n_clusters}, noise points: {n_noise}")
    if hasattr(clusterer, "probabilities_"):
        print("Avg membership probability (non-noise):", float(np.mean(clusterer.probabilities_[labels != -1])))

    plt.figure(figsize=(6, 5))
    plt.scatter(Xs[:, 0], Xs[:, 1], c=labels, s=18, cmap="tab10", alpha=0.9)
    plt.title("HDBSCAN (scaled): variable density blobs")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    print("\nNotes:")
    print("- HDBSCAN can mark uncertain points as noise (-1).")
    print("- min_cluster_size controls smallest cluster you consider meaningful.")
    print("- min_samples increases strictness (more points required to be in dense region).")


if __name__ == "__main__":
    main()
