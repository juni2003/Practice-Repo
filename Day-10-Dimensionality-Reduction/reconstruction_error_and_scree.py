"""
Reconstruction Error and Scree/Cumulative EVR

- Plot scree and cumulative EVR for choosing k
- Plot reconstruction MSE vs k
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error


def run():
    data = load_wine()
    X = data.data
    Xs = StandardScaler().fit_transform(X)

    # Fit full PCA
    pca_full = PCA(random_state=42).fit(Xs)
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)

    # Scree & cumulative EVR
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(1, len(evr) + 1), evr, "o-", lw=2)
    ax[0].set_title("Scree Plot (Wine)")
    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("EVR")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(np.arange(1, len(cum) + 1), cum, "o-", lw=2, color="green")
    ax[1].axhline(0.95, color="red", ls="--", label="95%")
    ax[1].set_title("Cumulative EVR (Wine)")
    ax[1].set_xlabel("Component")
    ax[1].set_ylabel("Cumulative EVR")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Reconstruction MSE vs k
    max_k = min(Xs.shape)
    mses = []
    ks = range(1, max_k + 1)
    for k in ks:
        pca = PCA(n_components=k, random_state=42).fit(Xs)
        Z = pca.transform(Xs)
        Xhat = pca.inverse_transform(Z)
        mses.append(mean_squared_error(Xs, Xhat))

    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), mses, "o-", lw=2)
    plt.title("Reconstruction MSE vs Components (Wine)")
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
