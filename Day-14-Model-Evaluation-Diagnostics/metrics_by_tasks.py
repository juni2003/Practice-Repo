"""
Metrics by Task: Classification, Regression, Ranking

- Classification: accuracy, precision, recall, f1, ROC-AUC
- Regression: MSE, MAE, R2
- Ranking: Average Precision (AP) and NDCG

Requirements: scikit-learn, numpy
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, ndcg_score
)


def classification_metrics():
    X, y = make_classification(n_samples=2000, n_features=15, n_informative=6, weights=[0.7, 0.3], random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    y_proba = pipe.predict_proba(Xte)[:, 1]

    print("Classification Metrics:")
    print("  Accuracy:", round(accuracy_score(yte, y_pred), 4))
    print("  Precision:", round(precision_score(yte, y_pred), 4))
    print("  Recall:", round(recall_score(yte, y_pred), 4))
    print("  F1:", round(f1_score(yte, y_pred), 4))
    print("  ROC-AUC:", round(roc_auc_score(yte, y_proba), 4))
    print("  AP (ranking metric):", round(average_precision_score(yte, y_proba), 4))


def regression_metrics():
    X, y = make_regression(n_samples=1500, n_features=10, n_informative=7, noise=20.0, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    y_pred = rf.predict(Xte)

    print("\nRegression Metrics:")
    print("  MSE:", round(mean_squared_error(yte, y_pred), 4))
    print("  MAE:", round(mean_absolute_error(yte, y_pred), 4))
    print("  R2:", round(r2_score(yte, y_pred), 4))


def ranking_metrics_demo():
    # Simple ranking scenario:
    # Relevance (ground truth) and predicted scores for a single query
    rel = np.array([[3, 2, 3, 0, 1]])  # relevance grades (higher is better)
    scores = np.array([[0.9, 0.7, 0.8, 0.1, 0.3]])  # predicted ranking scores
    k = 5
    print("\nRanking Metrics:")
    print("  NDCG@5:", round(ndcg_score(rel, scores, k=k), 4))
    # AP is demonstrated above in classification with average_precision_score


def main():
    classification_metrics()
    regression_metrics()
    ranking_metrics_demo()


if __name__ == "__main__":
    main()
