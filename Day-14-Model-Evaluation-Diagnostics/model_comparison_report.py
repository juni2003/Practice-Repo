"""
Model Comparison Report

- Dataset: breast cancer (binary classification)
- Models: LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Metrics: Accuracy, F1, ROC-AUC
- Stratified train/test split and uniform preprocessing (StandardScaler for LR only via Pipelines)

Requirements: scikit-learn, numpy, pandas (optional for pretty printing)
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate_model(name: str, clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    # Some models may not have predict_proba; guard accordingly
    try:
        y_proba = clf.predict_proba(Xte)[:, 1]
    except Exception:
        # Fallback via decision_function if available
        if hasattr(clf, "decision_function"):
            scores = clf.decision_function(Xte)
            # Min-max scale to [0,1] for AUC approximation
            smin, smax = scores.min(), scores.max()
            y_proba = (scores - smin) / (smax - smin + 1e-12)
        else:
            y_proba = y_pred.astype(float)

    return {
        "model": name,
        "accuracy": accuracy_score(yte, y_pred),
        "f1": f1_score(yte, y_pred),
        "roc_auc": roc_auc_score(yte, y_proba),
    }


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    models = [
        ("LogisticRegression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
        ])),
        ("RandomForest", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)),
    ]

    results = [evaluate_model(name, clf, Xtr, ytr, Xte, yte) for name, clf in models]
    print("Model Comparison (Breast Cancer):")
    for r in results:
        print(f"  {r['model']:>16} | Acc={r['accuracy']:.4f} | F1={r['f1']:.4f} | ROC-AUC={r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
