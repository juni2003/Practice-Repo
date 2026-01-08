"""
K-Means with scikit-learn — Practical Examples

What this file covers:
- Basic clustering with KMeans
- Standardization pipeline
- Elbow method and silhouette score (quick)
- MiniBatchKMeans for large datasets
- Cluster evaluation (when labels available) with ARI/NMI
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import Pipeline


def basic_kmeans():
    X, y_true = make_blobs(n_samples=700, centers=4, cluster_std=1.2, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42))
    ])
    labels = pipe.fit_predict(X)

    km: KMeans = pipe.named_steps["kmeans"]
    print("Inertia:", km.inertia_)
    print("Silhouette score:", silhouette_score(pipe.named_steps["scaler"].transform(X), labels))

    Xs = pipe.named_steps["scaler"].transform(X)
    centers = km.cluster_centers_
    plt.figure(figsize=(7, 6))
    plt.scatter(Xs[:, 0], Xs[:, 1], c=labels, cmap="tab10", s=15, alpha=0.8)
    plt.scatter(centers[:, 0], centers[:, 1], c="black", s=120, marker="X", label="Centroids")
    plt.title("K-Means (scaled) — clusters and centroids")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    # If true labels exist (here synthetic), evaluate with ARI/NMI
    print("Adjusted Rand Index:", adjusted_rand_score(y_true, labels))
    print("Normalized Mutual Info:", normalized_mutual_info_score(y_true, labels))


def elbow_and_silhouette_quick():
    X, _ = make_blobs(n_samples=600, centers=5, cluster_std=1.3, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    ks = range(2, 9)
    inertias = []
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs, labels))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(list(ks), inertias, "o-", lw=2)
    ax[0].set_title("Elbow: Inertia vs k")
    ax[0].set_xlabel("k")
    ax[0].set_ylabel("Inertia")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(list(ks), sils, "o-", lw=2, color="green")
    ax[1].set_title("Silhouette vs k")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Silhouette score")
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def minibatch_demo():
    X, _ = make_blobs(n_samples=10000, centers=6, cluster_std=1.4, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    mb = MiniBatchKMeans(n_clusters=6, batch_size=256, n_init=10, random_state=42)
    labels = mb.fit_predict(Xs)
    print("MiniBatch inertia:", mb.inertia_)

    # Compare to standard KMeans quickly (optional)
    km = KMeans(n_clusters=6, n_init=10, random_state=42)
    labels_km = km.fit_predict(Xs)
    print("Standard KMeans inertia:", km.inertia_)
    print("Silhouette (MiniBatch):", silhouette_score(Xs, labels))
    print("Silhouette (KMeans):", silhouette_score(Xs, labels_km))


if __name__ == "__main__":
    basic_kmeans()
    elbow_and_silhouette_quick()
    minibatch_demo()
