# PCA — Theory and Concepts

## Centering and Scaling
Given data matrix X ∈ R^{n×d} (n samples, d features):
1) Center: X_c = X − mean(X) (column-wise)
2) Optionally scale: divide by std if units/scales differ

## Covariance and Eigen-Decomposition
Covariance: C = (1/(n−1)) X_c^T X_c ∈ R^{d×d}

Eigen-decomposition:
C v_j = λ_j v_j,  with λ_1 ≥ λ_2 ≥ ... ≥ λ_d
- v_j: principal component (direction)
- λ_j: variance explained by component j

Explained variance ratio (EVR):
EVR_j = λ_j / Σ_k λ_k

## SVD Equivalence (Preferred Numerically)
X_c = U Σ V^T
- Right singular vectors (rows of V^T) are PCs
- Explained variance: λ_j = Σ_j^2 / (n−1)

## Projection and Reconstruction
Projection onto first k PCs:
Z_k = X_c V_k             # V_k ∈ R^{d×k}

Reconstruction:
X̂_k = Z_k V_k^T + mean

Reconstruction error (MSE):
MSE_k = (1/(n·d)) ||X − X̂_k||_F^2
Decreases as k increases (monotonic).

## Scree Plot and Cumulative EVR
- Scree plot: λ_j (or EVR_j) vs component index
- Cumulative EVR: choose k achieving threshold (e.g., 0.95)

## Whitening
Whitened PCs:
Z_w = Z_k Λ_k^{−1/2}      # Λ_k = diag(λ_1,...,λ_k)
Properties:
- Uncorrelated, unit variance along each PC
- Can benefit distance-based models; may harm others if scale is informative

## Incremental and Randomized PCA
- IncrementalPCA: processes data in batches (low-memory streaming)
- PCA(svd_solver="randomized"): faster on large, tall matrices for top components

## When Not to Use
- Categorical data without meaningful order
- Heavy nonlinear manifolds (consider t-SNE/UMAP for viz)
- When interpretability of original features is crucial

## Practical Notes
- Always fit PCA on training set only (avoid leakage)
- Scale before PCA for mixed-scale features
- Outliers can dominate PCs — consider robust scaling or trimming
