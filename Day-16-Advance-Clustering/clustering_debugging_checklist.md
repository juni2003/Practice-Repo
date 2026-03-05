# Advanced Clustering — Practical Debugging Checklist

## 1) Data prep (most common failure point)
- Scale numeric features (StandardScaler / RobustScaler).
- Remove extreme outliers if they dominate distances (or use robust scaling).
- If high dimensional (e.g., 1000+ features): consider PCA/UMAP before clustering.

## 2) DBSCAN / HDBSCAN issues
- If **everything is noise (-1)**:
  - Increase `eps` (DBSCAN) or decrease `min_samples`
  - Ensure data is scaled
- If **everything becomes one cluster**:
  - Decrease `eps`
  - Increase `min_samples`
- Different density clusters:
  - DBSCAN may fail → prefer HDBSCAN

## 3) GMM issues
- GMM expects ellipsoidal Gaussian-like clusters.
- If clusters are clearly non-convex, GMM isn’t the right tool.
- Choose number of components with BIC/AIC.
- Use soft assignments to inspect ambiguity (max responsibility).

## 4) Spectral clustering issues
- Sensitive to graph parameters:
  - Too small `n_neighbors` → graph disconnects → unstable clusters
  - Too large `n_neighbors` → loses locality → behaves more like KMeans
- Not ideal for very large datasets (eigen decomposition cost).

## 5) Validation sanity checks
- Don’t trust silhouette alone.
- Add stability checks (bootstrap ARI) to measure reproducibility.
- Inspect clusters qualitatively with 2D projections (PCA/TSNE/UMAP).

## 6) Actionable workflow
1. Scale features
2. Start with KMeans (baseline) + metrics
3. Try DBSCAN (outliers + non-convex)
4. If density varies → HDBSCAN
5. If overlap and uncertainty matters → GMM + BIC
6. If non-convex shapes → Spectral
7. Evaluate: internal metrics + stability + visual inspection
