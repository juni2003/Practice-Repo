"""
Scaling and Whitening Effects

- Show why scaling before PCA is important when features have different scales
- Show PCA(whiten=True) behavior and its impact on downstream classification

Notes:
- Whitening decorrelates and scales PCs to unit variance
- Helpful for some distance-based models; can hurt models relying on original scale
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def digits_logreg_with_without_whitening():
    X, y = load_digits(return_X_y=True)

    # Baseline: Standardize + PCA (no whitening) + LogisticRegression
    pipe_no_white = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=40, whiten=False, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    # Whitening variant
    pipe_white = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=40, whiten=True, random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None))
    ])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    acc_no_white = pipe_no_white.fit(Xtr, ytr).score(Xte, yte)
    acc_white = pipe_white.fit(Xtr, ytr).score(Xte, yte)

    print(f"Digits LogisticRegression with PCA(n=40):")
    print(f"  No whitening:  acc={acc_no_white:.4f}")
    print(f"  Whitening   :  acc={acc_white:.4f}")


def scaling_matter_demo():
    # Two features with very different variance
    rng = np.random.default_rng(42)
    n = 1000
    x1 = rng.normal(0, 1.0, size=n)
    x2 = rng.normal(0, 100.0, size=n)  # much larger scale
    X = np.c_[x1, x2]

    # PCA on unscaled data points PC towards x2 axis
    pca_unscaled = PCA(n_components=2, random_state=42).fit(X)
    # PCA on scaled data balances importance
    Xs = StandardScaler().fit_transform(X)
    pca_scaled = PCA(n_components=2, random_state=42).fit(Xs)

    print("Unscaled EVR:", np.round(pca_unscaled.explained_variance_ratio_, 4))
    print("Scaled   EVR:", np.round(pca_scaled.explained_variance_ratio_, 4))
    print("Unscaled components (rows):\n", np.round(pca_unscaled.components_, 4))
    print("Scaled   components (rows):\n", np.round(pca_scaled.components_, 4))


if __name__ == "__main__":
    scaling_matter_demo()
    digits_logreg_with_without_whitening()
