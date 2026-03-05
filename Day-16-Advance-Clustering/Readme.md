# Day 16 — Advanced Clustering

This module covers **advanced clustering algorithms** and—more importantly—**how to validate clustering properly** (because clustering has no labels by default).

You’ll learn and run code for:

- **DBSCAN** (density-based clustering with outliers)
- **HDBSCAN** (hierarchical DBSCAN; more robust than DBSCAN) *(optional dependency)*
- **Gaussian Mixture Models (GMM)** for **soft clustering** (probabilistic assignments)
- **Spectral Clustering** for **non-convex shapes** (e.g., “two moons”)
- **Cluster validation beyond silhouette**
  - internal metrics (silhouette, Calinski–Harabasz, Davies–Bouldin)
  - **stability via resampling** (how consistent are clusters?)

---

## Folder Structure

```
Day-16-Advanced-Clustering/
├── README.md
├── dbscan_hyperparams_and_outliers.py
├── hdbscan_intro_and_cluster_tree.py
├── gaussian_mixture_soft_clustering.py
├── spectral_clustering_nonconvex.py
├── cluster_validation_metrics_and_stability.py
└── clustering_debugging_checklist.md
```

---

## Install

Base (everything except HDBSCAN):
```bash
pip install numpy scikit-learn matplotlib
```

Optional (HDBSCAN script only):
```bash
pip install hdbscan
```

---

## How to Run

```bash
python Day-16-Advanced-Clustering/dbscan_hyperparams_and_outliers.py
python Day-16-Advanced-Clustering/gaussian_mixture_soft_clustering.py
python Day-16-Advanced-Clustering/spectral_clustering_nonconvex.py
python Day-16-Advanced-Clustering/cluster_validation_metrics_and_stability.py
python Day-16-Advanced-Clustering/hdbscan_intro_and_cluster_tree.py
```

---

## Key Ideas (Logic + Concepts)

### 1) DBSCAN (Density-Based Spatial Clustering)
DBSCAN defines clusters as **dense regions** separated by **low-density gaps**.

Parameters:
- `eps`: neighborhood radius
- `min_samples`: points needed in the neighborhood for a “core” point

Point types:
- **core**: has at least `min_samples` within `eps`
- **border**: near a core point but not dense enough itself
- **noise/outlier**: not reachable from any core → labeled as `-1`

DBSCAN is great when:
- clusters are irregular shapes
- you want outlier detection
- you don’t want to choose `k`

But it struggles when:
- clusters have different densities
- feature scaling isn’t handled

### 2) HDBSCAN
HDBSCAN builds a hierarchy of density-based clusters and extracts “stable” clusters.
It usually works better than DBSCAN when cluster densities vary.

Main parameter:
- `min_cluster_size`

It outputs:
- labels (with `-1` noise)
- cluster membership strengths (depending on settings)

### 3) Gaussian Mixture Models (GMM)
GMM assumes data comes from a mixture of Gaussians.
Unlike KMeans (hard assignments), GMM gives:
- `predict_proba(X)` → **soft assignments** (probabilities for each cluster)

Useful when:
- clusters overlap
- you want uncertainty of assignment

Model selection:
- use **AIC/BIC** to pick number of components

### 4) Spectral Clustering
Spectral clustering:
1. builds a similarity graph (who is near who)
2. computes a graph Laplacian
3. uses eigenvectors to embed points
4. clusters in that embedded space

Great for **non-convex** shapes (two moons / circles).

### 5) Validation beyond silhouette
Silhouette alone can be misleading, especially with:
- noise points
- varying densities
- non-convex structures

We also compute:
- Calinski–Harabasz (higher better)
- Davies–Bouldin (lower better)
- Stability via bootstrap/resampling:
  “If I sample the dataset again, do I get similar clusters?”

---

If you want, I can adapt Day 16 scripts to your own dataset (CSV) and show how to select `eps`, `min_samples`, and validate stability on real data.
