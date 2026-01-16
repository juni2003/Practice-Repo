"""
PCA from Scratch (via SVD)

Features:
- Fit: centers X, computes SVD, components, explained variance (EV), EV ratio
- Transform / inverse_transform
- Scree & cumulative EVR plot
- Reconstruction error curve

Educational; prefer sklearn.decomposition.PCA for production.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler


@dataclass
class PCAFromScratch:
    n_components: Optional[int] = None
    mean_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None  # shape (k, d)
    singular_values_: Optional[np.ndarray] = None  # shape (min(n,d),)
    explained_variance_: Optional[np.ndarray] = None
    explained_variance_ratio_: Optional[np.ndarray] = None
    fitted_: bool = False

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        X = np.asarray(X, dtype=float)
        n, d = X.shape

        # center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # SVD
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U S Vt
        # components are rows of Vt (shape (min(n,d), d)), top-first
        if self.n_components is None:
            k = min(n, d)
        else:
            k = int(self.n_components)
        self.components_ = Vt[:k, :]

        # explained variance λ_j = S_j^2 / (n-1)
        ev = (S**2) / (n - 1)
        self.singular_values_ = S[:k]
        self.explained_variance_ = ev[:k]
        self.explained_variance_ratio_ = ev[:k] / ev.sum()
        self.fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "Call fit() first."
        Xc = X - self.mean_
        Z = Xc @ self.components_.T
        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        assert self.fitted_, "Call fit() first."
        Xhat = Z @ self.components_ + self.mean_
        return Xhat

    def plot_scree(self, save_path: Optional[str] = None) -> None:
        evr = self.explained_variance_ratio_
        cum = np.cumsum(evr)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(np.arange(1, len(evr) + 1), evr, "o-", lw=2)
        ax[0].set_title("Scree: Explained Variance Ratio")
        ax[0].set_xlabel("Component")
        ax[0].set_ylabel("EVR")
        ax[0].grid(True, alpha=0.3)

        ax[1].plot(np.arange(1, len(cum) + 1), cum, "o-", lw=2, color="green")
        ax[1].axhline(0.95, color="red", ls="--", label="95%")
        ax[1].set_title("Cumulative EVR")
        ax[1].set_xlabel("Component")
        ax[1].set_ylabel("Cumulative EVR")
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def reconstruction_error_curve(X: np.ndarray, max_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns arrays (k_values, mse_values) for reconstruction error across k.
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if max_k is None:
            max_k = min(n, d)

        # Fit full PCA once to reuse components/means
        pca_full = PCAFromScratch(n_components=max_k).fit(X)
        Xc = X - pca_full.mean_
        mses = []
        ks = np.arange(1, max_k + 1)
        for k in ks:
            Vk = pca_full.components_[:k, :]
            Zk = Xc @ Vk.T
            Xhat = Zk @ Vk + pca_full.mean_
            mse = np.mean((X - Xhat) ** 2)
            mses.append(mse)
        return ks, np.array(mses)


def demo():
    data = load_wine()
    X = data.data
    X = StandardScaler().fit_transform(X)

    pca = PCAFromScratch(n_components=10).fit(X)
    print("Top-5 EVR:", np.round(pca.explained_variance_ratio_[:5], 4))
    pca.plot_scree()

    # Reconstruction error curve
    ks, mses = PCAFromScratch.reconstruction_error_curve(X, max_k=13)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, mses, "o-", lw=2)
    plt.title("Reconstruction MSE vs Components (Wine)")
    plt.xlabel("k")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()
