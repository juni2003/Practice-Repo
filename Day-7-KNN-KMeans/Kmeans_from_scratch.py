"""
K-Means Clustering — From Scratch

What this file covers:
- Objective: minimize within-cluster sum of squares (inertia)
- Initialization: random vs k-means++
- Iteration: assign → update → converge
- Convergence: tolerance or max_iter
- Example and visualization

Note: For real projects, prefer sklearn.cluster.KMeans or MiniBatchKMeans.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def init_centroids_kpp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    k-means++ initialization: pick first centroid randomly, then subsequent
    with probability proportional to squared distance to nearest centroid.
    """
    n_samples = X.shape[0]
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)

    # Choose first centroid
    idx = rng.integers(0, n_samples)
    centroids[0] = X[idx]

    # Distances to closest centroid
    d2 = np.full(n_samples, np.inf)
    for c in range(1, k):
        # Update d2 with nearest centroid distances
        d2 = np.minimum(d2, np.sum((X - centroids[c - 1]) ** 2, axis=1))
        # Choose next centroid weighted by d2
        probs = d2 / d2.sum()
        idx = rng.choice(n_samples, p=probs)
        centroids[c] = X[idx]
    return centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    init: str = "k-means++",
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Returns:
        labels: (n_samples,) cluster assignments
        centroids: (k, n_features)
        inertia: float (sum of squared distances to centroids)
        n_iter: iterations performed
    """
    rng = np.random.default_rng(random_state)

    if init == "k-means++":
        centroids = init_centroids_kpp(X, k, rng)
    elif init == "random":
        idx = rng.choice(X.shape[0], size=k, replace=False)
        centroids = X[idx].copy()
    else:
        raise ValueError("init must be 'k-means++' or 'random'")

    for it in range(max_iter):
        # Assign step
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n, k)
        labels = np.argmin(distances, axis=1)

        # Update step
        new_centroids = centroids.copy()
        for c in range(k):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = X[mask].mean(axis=0)
            else:
                # Handle empty cluster: reinitialize to a random point
                new_centroids[c] = X[rng.integers(0, X.shape[0])]

        # Check convergence
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    # Compute inertia
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    inertia = np.sum(distances**2)

    return labels, centroids, inertia, it + 1


def demo_blobs(k: int = 4) -> None:
    X, _ = make_blobs(n_samples=600, centers=k, cluster_std=1.3, random_state=42)
    X = StandardScaler().fit_transform(X)

    labels, centers, inertia, n_iter = kmeans(X, k=k, init="k-means++", random_state=42)

    plt.figure(figsize=(7, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=120, marker="X", label="Centroids")
    plt.title(f"K-Means (from scratch): k={k} | Inertia={inertia:.1f} | iters={n_iter}")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_blobs(k=4)
