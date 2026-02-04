"""
Reproducibility & Search Spaces

- Demonstrates consistent seeding across NumPy and scikit-learn
- Shows how RandomizedSearchCV random_state affects candidate sampling
- Provides guidelines for designing realistic search spaces

Outputs best params and shows that repeated runs with fixed seeds produce consistent results.
"""

from __future__ import annotations
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def run_once(seed=42):
    # Global seed
    np.random.seed(seed)

    X, y = make_classification(
        n_samples=3000, n_features=25, n_informative=10, n_redundant=5,
        class_sep=1.0, random_state=seed
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("rf", RandomForestClassifier(random_state=seed, n_jobs=-1))
    ])

    # Search space guidelines:
    # - Use log-scale for multiplicative hyperparams (e.g., n_estimators, C)
    # - Constrain depth/leaves to sensible ranges to prevent overfitting
    # - Subsample fractions typically in [0.5, 1.0]
    params = {
        "rf__n_estimators": np.random.randint(100, 600, size=50).tolist(),
        "rf__max_depth": [None] + list(np.arange(3, 15)),
        "rf__min_samples_split": [2, 5, 10, 20],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", 0.7, 0.9],
        "rf__bootstrap": [True, False],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rs = RandomizedSearchCV(
        pipe, params, n_iter=25, cv=cv, scoring="roc_auc", n_jobs=-1, random_state=seed, verbose=0
    )
    rs.fit(Xtr, ytr)
    y_proba = rs.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, y_proba)
    return rs.best_params_, rs.best_score_, auc


def main():
    bp1, cv1, auc1 = run_once(seed=42)
    bp2, cv2, auc2 = run_once(seed=42)

    print("Run 1:")
    print("  Best params:", bp1)
    print("  Best CV ROC-AUC:", round(cv1, 4))
    print("  Test ROC-AUC:", round(auc1, 4))

    print("\nRun 2 (same seed):")
    print("  Best params:", bp2)
    print("  Best CV ROC-AUC:", round(cv2, 4))
    print("  Test ROC-AUC:", round(auc2, 4))

    print("\nNote: With fixed seeds, RandomizedSearchCV samples the same candidates and CV shuffles identically.")


if __name__ == "__main__":
    main()
