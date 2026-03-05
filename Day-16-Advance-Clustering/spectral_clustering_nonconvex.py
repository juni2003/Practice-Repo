"""
Spectral Clustering for non-convex shapes

Spectral clustering:
1) builds a similarity graph (neighbors)
2) computes eigenvectors of a graph Laplacian
3) clusters in that embedded space

Works well on shapes like:
- two moons
- concentric circles

This script compares:
- KMeans (often fails on two moons)
- SpectralClustering (often succeeds)

Requirements: numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering


def main() -> None:
    X, _ = make_moons(n_samples=1000, noise=0.08, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    km_labels = km.fit_predict(Xs)

    spec = SpectralClustering(
        n_clusters=2,
        affinity="nearest_neighbors",
        n_neighbors=15,
        assign_labels="kmeans",
        random_state=42,
    )
    spec_labels = spec.fit_predict(Xs)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=km_labels, s=16, cmap="tab10")
    plt.title("KMeans on Two Moons")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=spec_labels, s=16, cmap="tab10")
    plt.title("Spectral Clustering on Two Moons")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    print("Notes:")
    print("- KMeans assumes convex/spherical clusters in Euclidean space.")
    print("- Spectral clustering uses a graph representation and can separate non-convex shapes.")
    print("- n_neighbors controls the graph connectivity (too low -> disconnected; too high -> loses local structure).")


if __name__ == "__main__":
    main()
