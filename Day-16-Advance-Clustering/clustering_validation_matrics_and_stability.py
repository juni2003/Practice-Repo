"""
Cluster Validation beyond silhouette:
- internal metrics: silhouette, calinski-harabasz, davies-bouldin
- stability via bootstrap resampling

Because clustering is unsupervised, we validate using:
1) internal cohesion/separation metrics
2) stability: do we get similar clusters under small data perturbations?

Stability approach (practical and simple):
- Run clustering on full data to get reference labels.
- Bootstrap sample rows multiple times.
- Cluster the sample.
- Compare sample labels (mapped back) to reference using Adjusted Rand Index (ARI).

Limitations:
- Mapping labels between runs is non-trivial for arbitrary algorithms.
- ARI is label-permutation invariant, but requires same set of points.

To handle bootstrap, we compare on the overlapping points between sample and full set.

Requirements: numpy, scikit-learn
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score


def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    # Some metrics require >=2 clusters and no all-noise case
    unique = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    if n_clusters < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}

    # Silhouette is not well-defined for noise label -1 in a pure way.
    # We'll compute it on non-noise points only for DBSCAN-like methods.
    if -1 in unique:
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
        X_use, y_use = X[mask], labels[mask]
    else:
        X_use, y_use = X, labels

    return {
        "silhouette": float(silhouette_score(X_use, y_use)),
        "calinski_harabasz": float(calinski_harabasz_score(X_use, y_use)),
        "davies_bouldin": float(davies_bouldin_score(X_use, y_use)),
    }


def stability_via_bootstrap(X: np.ndarray, cluster_fn, n_boot: int = 40, seed: int = 42) -> float:
    """
    cluster_fn: callable that takes X and returns labels for X
    Returns mean ARI to reference clustering on full X (over bootstrap samples).
    """
    rng = np.random.default_rng(seed)

    ref_labels = cluster_fn(X)
    aris = []

    n = len(X)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap indices with replacement
        # Need unique indices to compare ARI on the same points
        uniq = np.unique(idx)
        Xb = X[uniq]
        labels_b = cluster_fn(Xb)
        # Compare labels of the same points:
        aris.append(adjusted_rand_score(ref_labels[uniq], labels_b))

    return float(np.mean(aris))


def main() -> None:
    X, _ = make_blobs(n_samples=1300, centers=4, cluster_std=[0.6, 0.8, 0.5, 1.0], random_state=42)
    Xs = StandardScaler().fit_transform(X)

    # Define clusterers as callables (for stability function)
    kmeans = lambda Z: KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(Z)
    dbscan = lambda Z: DBSCAN(eps=0.25, min_samples=10).fit_predict(Z)

    for name, fn in [("KMeans(k=4)", kmeans), ("DBSCAN(eps=0.25)", dbscan)]:
        labels = fn(Xs)
        mets = internal_metrics(Xs, labels)
        stab = stability_via_bootstrap(Xs, fn, n_boot=35, seed=42)

        unique = set(labels)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        n_noise = int(np.sum(labels == -1))

        print(f"\n{name}")
        print(f"  clusters={n_clusters}, noise={n_noise}")
        print(f"  silhouette={mets['silhouette']:.4f}")
        print(f"  calinski_harabasz={mets['calinski_harabasz']:.2f}")
        print(f"  davies_bouldin={mets['davies_bouldin']:.4f}")
        print(f"  stability (mean ARI bootstrap)={stab:.4f}")

    print("\nHow to interpret:")
    print("- Silhouette/CH/DB measure compactness vs separation (internal only).")
    print("- Stability answers: are clusters reproducible under resampling?")
    print("- A model can have decent silhouette but poor stability (not reliable).")


if __name__ == "__main__":
    main()
