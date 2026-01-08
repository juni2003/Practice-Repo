# Day 7: K-Nearest Neighbors (KNN) and K-Means Clustering

## Overview
This module covers two fundamental distance-based algorithms:
- K-Nearest Neighbors (KNN): Supervised learning for classification and regression
- K-Means: Unsupervised learning for clustering

Both rely on a notion of distance between points, so feature scaling and metric choice are critical.

## Why These Next?
- Simpler than SVMs but build strong intuition around distances
- KNN bridges supervised learning; K-Means bridges unsupervised learning
- Common in recommendation, segmentation, anomaly detection, and baselines

## What You’ll Learn
- KNN: Voting, distance metrics, scaling, choosing k, weighted neighbors
- K-Means: Objective function, initialization (k-means++), inertia, convergence
- Model selection: Elbow method, silhouette score
- Practical pipelines with scikit-learn
- Common mistakes and best practices

## Module Structure
```
Day-7-KNN-KMeans/
├── README.md
├── knn_from_scratch.py
├── kmeans_from_scratch.py
├── distance_metrics_and_scaling.py
├── knn_sklearn_examples.py
├── kmeans_sklearn_examples.py
├── elbow_silhouette_methods.py
└── common_mistakes_and_best_practices.txt
```

## Quick Start (KNN Classification)
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="minkowski", p=2))
])
pipe.fit(X_train, y_train)
print("Accuracy:", pipe.score(X_test, y_test))
```

## Quick Start (K-Means Clustering)
```python
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.2, random_state=42)
X = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=42)
labels = km.fit_predict(X)
print("Inertia:", km.inertia_)
print("Cluster centers shape:", km.cluster_centers_.shape)
```

## Key Concepts

### KNN (Classification)
- Predict class by majority vote among k nearest neighbors
- Weighted voting (e.g., inverse distance) reduces tie sensitivity and improves boundary smoothness
- Hyperparameters:
  - k (n_neighbors): odd number helps avoid ties in binary classification
  - metric (e.g., Euclidean, Manhattan)
  - weights: "uniform" vs "distance"
  - algorithm: "auto", "kd_tree", "ball_tree", "brute" (for performance)
- Sensitive to scaling and curse of dimensionality

### K-Means (Clustering)
- Objective: minimize within-cluster sum of squared distances (inertia)
- Iterative: assign points → update centroids → repeat until convergence
- Initialization matters: k-means++ recommended
- Choosing k:
  - Elbow method: kink in inertia vs k plot
  - Silhouette score: [-1,1], higher is better (cohesion vs separation)
- Assumes spherical clusters of similar size; sensitive to outliers and scaling

## Choosing Distance Metrics
- Euclidean (L2): default, smooth decision boundaries
- Manhattan (L1): robust in high-dim or axis-aligned distances
- Minkowski (p): generalization (p=1 → L1, p=2 → L2)
- Cosine: use for text or directional similarity (KMeans with cosine often via pre-processing)

## Common Pitfalls (See file for details)
- Not scaling features
- Wrong k selection
- Using K-Means for non-spherical clusters
- Ignoring class imbalance in KNN
- Using K-Means on categorical data (consider k-modes/k-prototypes)

## Practice Flow
1) Read theory in this README and skim common mistakes
2) Run knn_from_scratch.py and kmeans_from_scratch.py to learn internals
3) Explore distance_metrics_and_scaling.py to see scaling/metric effects
4) Use knn_sklearn_examples.py and kmeans_sklearn_examples.py for practical pipelines
5) Run elbow_silhouette_methods.py to choose k on your data
