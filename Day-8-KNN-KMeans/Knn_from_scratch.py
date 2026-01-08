"""
K-Nearest Neighbors (KNN) Classification — From Scratch

What this file covers:
- A simple, educational KNN classifier (classification)
- Metrics: Euclidean (L2), Manhattan (L1), Minkowski (p)
- Weights: uniform vs inverse distance
- Example on synthetic data with/without scaling

Note: For real projects, use sklearn.neighbors.KNeighborsClassifier.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional, Tuple
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pairwise_distance(
    X: np.ndarray,
    x: np.ndarray,
    metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
    p: int = 2
) -> np.ndarray:
    """
    Compute distances between each row in X and single vector x.
    """
    diff = X - x
    if metric == "euclidean":
        return np.sqrt(np.sum(diff**2, axis=1))
    elif metric == "manhattan":
        return np.sum(np.abs(diff), axis=1)
    elif metric == "minkowski":
        return np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


class KNNClassifierScratch:
    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "uniform",
        metric: Literal["euclidean", "manhattan", "minkowski"] = "euclidean",
        p: int = 2
    ):
        assert n_neighbors >= 1, "n_neighbors must be >= 1"
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifierScratch":
        self.X_ = np.asarray(X)
        self.y_ = np.asarray(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.X_ is not None and self.y_ is not None, "Call fit() first"
        X = np.asarray(X)
        preds = []
        for i in range(X.shape[0]):
            dists = pairwise_distance(self.X_, X[i], self.metric, self.p)
            nn_idx = np.argsort(dists)[: self.n_neighbors]
            nn_labels = self.y_[nn_idx]
            if self.weights == "uniform":
                vote = Counter(nn_labels).most_common(1)[0][0]
            else:
                # distance-weighted voting (w = 1 / (d + eps))
                eps = 1e-9
                w = 1.0 / (dists[nn_idx] + eps)
                # Sum weights per class
                class_weights = {}
                for label, weight in zip(nn_labels, w):
                    class_weights[label] = class_weights.get(label, 0.0) + weight
                vote = max(class_weights.items(), key=lambda kv: kv[1])[0]
            preds.append(vote)
        return np.array(preds)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return (y_pred == y).mean()


def demo_synthetic(scale: bool = True) -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.25,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    knn = KNNClassifierScratch(n_neighbors=7, weights="distance", metric="minkowski", p=2)
    knn.fit(X_train, y_train)

    acc = knn.score(X_test, y_test)
    print(f"Scaled={scale} | KNN test accuracy: {acc:.4f}")

    # Plot decision boundary
    h = 0.03
    x_min, x_max = X_train[:, 0].min() - 1.0, X_train[:, 0].max() + 1.0
    y_min, y_max = X_train[:, 1].min() - 1.0, X_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdYlBu", edgecolors="k", s=30)
    plt.title(f"KNN (k=7, weights=distance) | Scaled={scale}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("With scaling:")
    demo_synthetic(scale=True)
    print("\nWithout scaling (for comparison):")
    demo_synthetic(scale=False)
