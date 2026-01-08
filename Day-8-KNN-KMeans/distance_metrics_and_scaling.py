"""
Distance Metrics and Feature Scaling — Practical Effects

What this file covers:
- Visual impact of scaling on KNN decision boundaries
- Metric choice: Euclidean (L2) vs Manhattan (L1) vs Minkowski p
- Show before/after scaling on the same dataset
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


def plot_knn_boundary(X, y, title, n_neighbors=7, metric="minkowski", p=2):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric=metric, p=p if metric == "minkowski" else 2)
    knn.fit(X, y)

    h = 0.03
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=25)
    plt.title(title)
    plt.grid(True, alpha=0.2)


def main():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        class_sep=1.0,
        random_state=42
    )
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    # Unscaled
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plot_knn_boundary(X_train, y_train, "Unscaled | Euclidean (p=2)", n_neighbors=7, metric="minkowski", p=2)

    plt.subplot(2, 2, 2)
    plot_knn_boundary(X_train, y_train, "Unscaled | Manhattan (p=1)", n_neighbors=7, metric="minkowski", p=1)

    # StandardScaler
    Xs = StandardScaler().fit_transform(X_train)
    plt.subplot(2, 2, 3)
    plot_knn_boundary(Xs, y_train, "StandardScaled | Euclidean (p=2)", n_neighbors=7, metric="minkowski", p=2)

    # MinMaxScaler
    Xm = MinMaxScaler().fit_transform(X_train)
    plt.subplot(2, 2, 4)
    plot_knn_boundary(Xm, y_train, "MinMaxScaled | Euclidean (p=2)", n_neighbors=7, metric="minkowski", p=2)

    plt.suptitle("KNN Decision Boundaries — Scaling and Metric Effects", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
