"""
Model Selection for K-Means — Elbow and Silhouette

What this file covers:
- Elbow plot utility: inertia vs k
- Silhouette analysis: average scores and detailed silhouette plot
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


def elbow_plot(Xs: np.ndarray, ks=range(2, 11)) -> None:
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(Xs)
        inertias.append(km.inertia_)
    plt.figure(figsize=(7, 5))
    plt.plot(list(ks), inertias, "o-", lw=2)
    plt.title("Elbow Method: Inertia vs k")
    plt.xlabel("k")
    plt.ylabel("Inertia (Within-Cluster SSE)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def silhouette_analysis(Xs: np.ndarray, k: int) -> None:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    sil_avg = silhouette_score(Xs, labels)
    print(f"Average silhouette score for k={k}: {sil_avg:.4f}")

    sample_sil = silhouette_samples(Xs, labels)
    y_lower = 10
    plt.figure(figsize=(7, 6))
    for i in range(k):
        ith_sil = sample_sil[labels == i]
        ith_sil.sort()
        size_i = ith_sil.shape[0]
        y_upper = y_lower + size_i
        color = plt.cm.tab10(float(i) / k)
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, facecolor=color, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=sil_avg, color="red", linestyle="--", lw=2, label=f"Avg={sil_avg:.2f}")
    plt.title(f"Silhouette Plot (k={k})")
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X, _ = make_blobs(n_samples=800, centers=5, cluster_std=1.5, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    elbow_plot(Xs, ks=range(2, 10))
    silhouette_analysis(Xs, k=5)


if __name__ == "__main__":
    main()
