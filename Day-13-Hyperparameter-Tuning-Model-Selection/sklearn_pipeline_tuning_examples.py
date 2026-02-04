"""
Pipeline Tuning with ColumnTransformer (Mixed Data)

- Synthetic dataset: numeric + categorical columns
- Pipeline: ColumnTransformer(OneHotEncoder + passthrough) → LogisticRegression
- GridSearchCV with StratifiedKFold
- Prints best params and scores

Requires: pandas, scikit-learn
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def build_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50_000, 15_000, n),
        "city": rng.choice(["A", "B", "C", "D"], n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "visits": rng.integers(1, 20, n),
    })
    # Target influenced by age/income/device/city + noise
    score = (
        (df["age"] > 40).astype(int) +
        (df["income"] > 60000).astype(int) +
        (df["device"] == "desktop").astype(int) +
        (df["city"].isin(["A", "B"])).astype(int) +
        (df["visits"] > 10).astype(int) +
        rng.integers(0, 2, n)
    )
    y = (score > 2).astype(int).values
    return df, y


def main():
    df, y = build_dataset()
    cat_cols = ["city", "device"]
    num_cols = [c for c in df.columns if c not in cat_cols]

    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, stratify=y, random_state=42)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42))
    ])

    params = {
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__class_weight": [None, "balanced"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, params, cv=cv, scoring="roc_auc", n_jobs=-1)
    gs.fit(Xtr, ytr)

    y_proba = gs.predict_proba(Xte)[:, 1]
    print("Best params:", gs.best_params_)
    print("Best CV ROC-AUC:", round(gs.best_score_, 4))
    print("Test ROC-AUC:", round(roc_auc_score(yte, y_proba), 4))


if __name__ == "__main__":
    main()
