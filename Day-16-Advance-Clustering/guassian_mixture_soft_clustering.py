"""
Gaussian Mixture Models (GMM): soft assignments + model selection via BIC/AIC

GMM assumes data is generated from a mixture of Gaussian distributions.
Unlike KMeans (hard labels), GMM provides:
- responsibilities: P(cluster=k | x) via predict_proba

This script:
- generates overlapping Gaussian blobs
- fits multiple component counts
- prints BIC/AIC to choose K
- visualizes hard labels and "confidence" (max responsibility)

Requirements: numpy, scikit-learn, matplotlib
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture


def main() -> None:
    X, _ = make_blobs(
        n_samples=1200,
        centers=[(-2, -1), (1.2, 0.8), (3.0, -2.0)],
        cluster_std=[1.0, 1.2, 0.9],  # overlap on purpose
        random_state=42,
    )
    Xs = StandardScaler().fit_transform(X)

    ks = range(1, 8)
    bics, aics = [], []
    models = {}

    for k in ks:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(Xs)
        bics.append(gmm.bic(Xs))
        aics.append(gmm.aic(Xs))
        models[k] = gmm

    best_k_bic = int(ks[int(np.argmin(bics))])
    print("BIC by k:", {k: round(v, 2) for k, v in zip(ks, bics)})
    print("AIC by k:", {k: round(v, 2) for k, v in zip(ks, aics)})
    print(f"Best k by BIC: {best_k_bic}")

    gmm = models[best_k_bic]
    labels = gmm.predict(Xs)
    proba = gmm.predict_proba(Xs)
    confidence = proba.max(axis=1)  # soft assignment strength

    # Plot clustering results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=labels, s=18, cmap="tab10", alpha=0.9)
    plt.title(f"GMM Hard Labels (k={best_k_bic})")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.scatter(Xs[:, 0], Xs[:, 1], c=confidence, s=18, cmap="viridis", alpha=0.9)
    plt.colorbar(label="max responsibility")
    plt.title("GMM Soft Assignment Confidence")
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass

    print("\nInterpretation:")
    print("- In overlap regions, max responsibility is lower (model is uncertain).")
    print("- BIC/AIC helps choose number of components to balance fit vs complexity.")


if __name__ == "__main__":
    main()
