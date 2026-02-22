"""
Feature Selection: Univariate, RFE, Model-based selection (embedded)

We build a numeric dataset with many noisy features and compare:
1) Univariate filter: SelectKBest(mutual_info_classif)
2) Wrapper: RFE(LogisticRegression)
3) Embedded: SelectFromModel(L1 LogisticRegression)

All are evaluated with cross-validation.

Key idea:
- Do selection inside a Pipeline, otherwise you leak information from validation folds.
"""

from __future__ import annotations

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel


def evaluate(pipe, X, y, name: str) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    print(f"{name:>24} | CV ROC-AUC mean={scores.mean():.4f} std={scores.std():.4f}")


def main() -> None:
    X, y = make_classification(
        n_samples=3500,
        n_features=60,
        n_informative=10,
        n_redundant=10,
        n_repeated=0,
        class_sep=1.0,
        weights=[0.6, 0.4],
        flip_y=0.03,
        random_state=42,
    )

    base_clf = LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)

    # 1) Univariate selection
    univariate = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kbest", SelectKBest(score_func=mutual_info_classif, k=15)),
            ("clf", base_clf),
        ]
    )

    # 2) RFE (wrapper) — uses an estimator to rank features
    # Use a sparse-friendly solver for ranking stability
    rfe_est = LogisticRegression(max_iter=3000, solver="liblinear", random_state=42)
    rfe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("rfe", RFE(estimator=rfe_est, n_features_to_select=15, step=0.2)),
            ("clf", base_clf),
        ]
    )

    # 3) Embedded (model-based): L1 drives many coefficients to zero
    l1 = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sfm", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.3, max_iter=3000, random_state=42))),
            ("clf", base_clf),
        ]
    )

    # Baseline: no selection
    baseline = Pipeline(steps=[("scaler", StandardScaler()), ("clf", base_clf)])

    evaluate(baseline, X, y, "Baseline (no selection)")
    evaluate(univariate, X, y, "Univariate SelectKBest")
    evaluate(rfe, X, y, "RFE")
    evaluate(l1, X, y, "Model-based (L1 SFM)")

    print("\nNotes:")
    print("- Univariate is fast but ignores feature interactions.")
    print("- RFE can capture some interaction patterns but is slower (repeated fitting).")
    print("- L1 model-based selection is efficient and often works well when a sparse solution is reasonable.")


if __name__ == "__main__":
    main()
