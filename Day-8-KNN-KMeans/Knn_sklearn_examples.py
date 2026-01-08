"""
KNN with scikit-learn — Practical Examples

What this file covers:
- Classification with pipelines and scaling
- Hyperparameter tuning for k, weights, and metric
- KNN regression example
- Performance notes: algorithm selection (brute, kd_tree, ball_tree)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report


def classification_basic():
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    params = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2]  # Manhattan vs Euclidean
    }

    gs = GridSearchCV(pipe, params, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print("Best CV accuracy:", gs.best_score_)
    print("Test accuracy:", gs.score(X_test, y_test))
    y_pred = gs.predict(X_test)
    print("\nClassification report:\n", classification_report(y_test, y_pred))


def regression_basic():
    X, y = make_regression(n_samples=600, n_features=5, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor())
    ])

    params = {
        "knn__n_neighbors": [3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["minkowski"],
        "knn__p": [1, 2]
    }

    gs = GridSearchCV(pipe, params, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    print("Best params:", gs.best_params_)
    print("R2:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))


def performance_note():
    """
    Performance tip: algorithm parameter
    - 'brute' (default for most metrics) scans all points (OK for small to medium n)
    - 'kd_tree' and 'ball_tree' can accelerate queries for low to moderate dimensions
    """
    print("""
KNN Performance Tips:
- For high-dimensional data, brute-force can be as good as (or better than) trees.
- For low-dim (<= ~20), try algorithm='kd_tree' or 'ball_tree'.
- Always benchmark on your dataset.
""")


if __name__ == "__main__":
    classification_basic()
    regression_basic()
    performance_note()
