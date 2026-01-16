# Day 10: Dimensionality Reduction — PCA and Friends

## Overview
Principal Component Analysis (PCA) is a linear dimensionality reduction method that:
- Finds orthogonal directions (principal components) capturing maximum variance
- Projects data onto top-k components to compress/denoise/visualize
- Provides uncorrelated features and interpretable variance explained

This module covers:
- PCA intuition: variance, eigenvectors, scree plot
- Whitening, explained variance, reconstruction error
- Incremental PCA and randomized SVD (when and why)
- 2D/3D visualization on real datasets
- Best practices and when not to use PCA

## Why PCA?
- Compress high-dimensional data while preserving most signal
- Visualize data structure (2D/3D)
- Decorrelate features and reduce noise
- Speed up downstream models

Limitations:
- Linear only (can’t capture nonlinear structure)
- Components are linear combinations (less interpretable than original features)
- Sensitive to feature scaling and outliers

## Module Structure
```
Day-10-Dimensionality-Reduction/
├── README.md                                  # This file
├── pca_theory_and_concepts.md                 # Math: covariance/SVD, EVR, whitening
├── pca_from_scratch.py                        # Educational PCA via SVD
├── pca_sklearn_examples.py                    # PCA on Iris/Digits with 2D/3D visuals, biplot
├── incremental_pca_and_randomized_pca.py      # IncrementalPCA + randomized solver comparison
├── whitening_and_scaling_effects.py           # Scaling & whitening demos, impact on models
├── reconstruction_error_and_scree.py          # Scree/cumulative plots and reconstruction error
└── common_mistakes_and_best_practices.txt     # Pitfalls and tips
```

## Quick Start (Sklearn PCA)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative EVR:", pca.explained_variance_ratio_.sum())
```

## Key Concepts
- Centering and scaling
  - Center X by subtracting mean; scale when features have different units
- Variance maximization
  - Components are directions maximizing variance subject to orthogonality
- Eigen/SVD equivalence
  - PCA via covariance eigen-decomposition or directly via SVD of centered X
- Explained variance ratio (EVR)
  - Fraction of total variance captured by each component
- Scree and cumulative EVR plots
  - Choose k via elbow or target EVR (e.g., 95%)
- Whitening
  - Scales projected components to unit variance; decorrelates and normalizes PCs

## When NOT to use PCA
- Nonlinear structure dominates (try t-SNE/UMAP for visualization)
- Features are categorical without meaningful ordering
- You need fully interpretable original features
- Leakage risk: fitting PCA on full dataset before split

## Typical Workflows
- Compression: choose k via cumulative EVR (e.g., 90–99%)
- Visualization: choose k = 2 or 3 and color by label/cluster
- Preprocessing: PCA → model (avoid over-whitening if model needs scale)

## Files in this module
- pca_from_scratch.py: Fit/transform/inverse_transform with SVD; scree and reconstruction error
- pca_sklearn_examples.py: Iris (biplot), Digits (2D/3D), EVR curves
- incremental_pca_and_randomized_pca.py: Memory/performance trade-offs
- whitening_and_scaling_effects.py: Impact of scaling and PCA(whiten=True) on models
- reconstruction_error_and_scree.py: Pick k via error/EVR, save plots
- common_mistakes_and_best_practices.txt: Avoid pitfalls

Happy reducing!
