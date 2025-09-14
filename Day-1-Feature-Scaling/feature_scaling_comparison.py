"""
Feature Scaling Comparison (Main)

This script compares model performance with and without common feature scaling
techniques highlighted in the README:
- Standardization (Z-score)
- Min-Max Scaling
- Robust Scaling

Models chosen are sensitive to feature scales:
- K-Nearest Neighbors (distance-based)
- Support Vector Machine with RBF kernel

Usage:
    python feature_scaling_comparison.py
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

def evaluate_scalers(
    X: np.ndarray,
    y: np.ndarray,
    estimators: Dict[str, object],
    scalers: Dict[str, Optional[object]],
    cv: StratifiedKFold,
    scoring: str = "accuracy",
) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Returns dict keyed by (estimator_name, scaler_name) -> (mean, std)
    """
    results: Dict[Tuple[str, str], Tuple[float, float]] = {}

    for est_name, estimator in estimators.items():
        for sc_name, scaler in scalers.items():
            if scaler is None:
                steps = [("clf", estimator)]
            else:
                steps = [("scaler", scaler), ("clf", estimator)]

            pipe = Pipeline(steps)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None)
            results[(est_name, sc_name)] = (scores.mean(), scores.std())
    return results

def main() -> None:
    # Dataset with continuous features that benefits from scaling
    data = load_breast_cancer()
    X, y = data.data, data.target

    estimators = {
        "KNN(k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM(RBF)": SVC(kernel="rbf", gamma="scale", C=1.0),
    }

    scalers = {
        "None": None,  # baseline
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = evaluate_scalers(X, y, estimators, scalers, cv=cv, scoring="accuracy")

    # Pretty print
    est_names = list(estimators.keys())
    sc_names = list(scalers.keys())

    print("\nFeature Scaling Comparison (5-fold CV Accuracy)")
    print("-" * 70)
    header = ["Estimator"] + sc_names
    print("{:15} ".format(header[0]) + " | ".join(f"{h:<15}" for h in header[1:]))
    print("-" * 70)

    for est in est_names:
        row = [est]
        for sc in sc_names:
            mean, std = results[(est, sc)]
            row.append(f"{mean:.4f} ± {std:.4f}")
        print("{:15} ".format(row[0]) + " | ".join(f"{c:<15}" for c in row[1:]))

    print("\nNotes:")
    print("- Distance and kernel methods (KNN, SVM) typically improve with scaling.")
    print("- Standardization is often a strong default; RobustScaler helps with outliers.")
    print("- MinMaxScaler is useful when a bounded [0,1] range is desired.\n")


if __name__ == "__main__":
    main()