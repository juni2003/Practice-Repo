"""
Incremental PCA and Randomized PCA

- IncrementalPCA: partial_fit batches to handle large data (low memory)
- PCA(svd_solver="randomized"): faster approximate SVD for top components

Demos:
- Compare timings roughly (small demonstration scale)
- Validate that EVR and recon errors are similar for moderate k
"""

from __future__ import annotations
import numpy as np
import time
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def demo_incremental_vs_full(n=5000, d=200, k=50, batch_size=500, random_state=42):
    X, _ = make_classification(
        n_samples=n, n_features=d, n_informative=int(d * 0.6),
        random_state=random_state
    )
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # Full PCA (randomized for speed)
    t0 = time.time()
    pca_full = PCA(n_components=k, svd_solver="randomized", random_state=random_state).fit(X)
    Z_full = pca_full.transform(X)
    Xhat_full = pca_full.inverse_transform(Z_full)
    t_full = time.time() - t0
    mse_full = mean_squared_error(X, Xhat_full)
    evr_full = pca_full.explained_variance_ratio_.sum()

    # Incremental PCA
    ipca = IncrementalPCA(n_components=k)
    t0 = time.time()
    for i in range(0, X.shape[0], batch_size):
        ipca.partial_fit(X[i:i+batch_size])
    Z_inc = ipca.transform(X)
    Xhat_inc = ipca.inverse_transform(Z_inc)
    t_inc = time.time() - t0
    mse_inc = mean_squared_error(X, Xhat_inc)
    evr_inc = (ipca.explained_variance_ratio_).sum()

    print(f"Full PCA (randomized): time={t_full:.2f}s, cumEVR={evr_full:.4f}, MSE={mse_full:.4f}")
    print(f"Incremental PCA     : time={t_inc:.2f}s, cumEVR={evr_inc:.4f}, MSE={mse_inc:.4f}")


if __name__ == "__main__":
    demo_incremental_vs_full()
