"""
PCA with scikit-learn — Visualizations and Biplot

Demos:
- Iris 2D projection + biplot of loadings
- Digits 2D and 3D visualization
- Explained variance curves
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def iris_2d_biplot():
    iris = load_iris()
    X, y = iris.data, iris.target
    names = iris.target_names
    features = iris.feature_names

    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    # Scatter
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(y)):
        plt.scatter(Z[y == label, 0], Z[y == label, 1], label=names[label], alpha=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.title("Iris — PCA 2D Projection")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Biplot arrows for feature loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    for i, feat in enumerate(features):
        plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color="r", alpha=0.6, head_width=0.05)
        plt.text(loadings[i, 0]*1.1, loadings[i, 1]*1.1, feat, color="r", fontsize=9)

    plt.tight_layout()
    plt.show()


def digits_2d_3d():
    digits = load_digits()
    X, y = digits.data, digits.target

    Xs = StandardScaler().fit_transform(X)

    # 2D
    pca2 = PCA(n_components=2, random_state=42)
    Z2 = pca2.fit_transform(Xs)
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=y, cmap="tab10", s=15)
    plt.colorbar(sc, label="Digit")
    plt.title("Digits — PCA 2D")
    plt.xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 3D
    pca3 = PCA(n_components=3, random_state=42)
    Z3 = pca3.fit_transform(Xs)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], c=y, cmap="tab10", s=10)
    fig.colorbar(p, ax=ax, label="Digit")
    ax.set_title("Digits — PCA 3D")
    ax.set_xlabel(f"PC1 ({pca3.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca3.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca3.explained_variance_ratio_[2]*100:.1f}%)")
    plt.tight_layout()
    plt.show()


def explained_variance_curves():
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data)

    pca = PCA(random_state=42).fit(X)  # all comps
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(np.arange(1, len(evr)+1), evr, "o-", lw=2)
    ax[0].set_title("Scree Plot (Digits)")
    ax[0].set_xlabel("Component")
    ax[0].set_ylabel("EVR")
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(np.arange(1, len(cum)+1), cum, "o-", lw=2, color="green")
    ax[1].axhline(0.95, color="red", ls="--", label="95%")
    ax[1].set_title("Cumulative EVR (Digits)")
    ax[1].set_xlabel("Component")
    ax[1].set_ylabel("Cumulative EVR")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    iris_2d_biplot()
    digits_2d_3d()
    explained_variance_curves()
