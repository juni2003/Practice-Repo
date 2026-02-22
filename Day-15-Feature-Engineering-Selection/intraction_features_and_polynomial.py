"""
Interaction Features

This script demonstrates how interaction features help a linear model
when the true signal contains interactions.

We generate data where the label depends on x1 * x2.
- LogisticRegression WITHOUT interaction terms struggles.
- LogisticRegression WITH PolynomialFeatures(interaction_only=True) improves.

Key logic:
- Interaction expansion is a preprocessing step.
- Keep it inside the Pipeline to avoid leakage in CV.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression


def make_interaction_data(n: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    # True signal depends on interaction between feature 0 and 1
    interaction = X[:, 0] * X[:, 1]
    logits = 1.5 * interaction + 0.2 * X[:, 2] - 0.1 * X[:, 3]
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)
    return X, y


def eval_pipe(pipe, X, y, name: str):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    print(f"{name:>28} | ROC-AUC mean={scores.mean():.4f} std={scores.std():.4f}")


def main() -> None:
    X, y = make_interaction_data()

    base = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
        ]
    )

    with_interactions = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("scaler", StandardScaler(with_mean=False)),  # safe even if poly returns sparse in some cases
            ("clf", LogisticRegression(max_iter=3000, solver="lbfgs", random_state=42)),
        ]
    )

    eval_pipe(base, X, y, "Linear model (no interactions)")
    eval_pipe(with_interactions, X, y, "Linear model (+ interactions)")

    print("\nInterpretation:")
    print("- If the true pattern depends on x1*x2, a plain linear model can't represent it.")
    print("- Adding interaction features lets the linear model capture that dependency.")


if __name__ == "__main__":
    main()
