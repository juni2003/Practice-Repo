"""
ColumnTransformer Pipelines (mixed numeric + categorical)

This script builds a synthetic dataset with:
- Numeric columns: age, income, visits
- Categorical columns: city, device
Target: binary classification

It demonstrates:
- ColumnTransformer with (imputer + scaler) for numeric
- OneHotEncoder for categorical
- Full Pipeline end-to-end
- Cross-validated ROC-AUC evaluation

Key logic:
- Preprocessing is fit ONLY on training folds when used inside cross_val_score,
  preventing leakage automatically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def make_data(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "age": rng.integers(18, 70, n),
            "income": rng.normal(50_000, 15_000, n),
            "visits": rng.integers(1, 30, n),
            "city": rng.choice(["A", "B", "C", "D"], n, p=[0.4, 0.3, 0.2, 0.1]),
            "device": rng.choice(["mobile", "desktop", "tablet"], n, p=[0.6, 0.3, 0.1]),
        }
    )

    # Inject a little missingness to show imputers matter
    miss_mask = rng.random(n) < 0.05
    df.loc[miss_mask, "income"] = np.nan

    # Construct a target with some real signal + noise
    signal = (
        0.8 * (df["age"] > 40).astype(int)
        + 0.9 * (df["income"].fillna(df["income"].median()) > 60_000).astype(int)
        + 0.7 * (df["device"] == "desktop").astype(int)
        + 0.4 * (df["city"].isin(["A", "B"])).astype(int)
        + 0.5 * (df["visits"] > 12).astype(int)
    )
    # Convert to probability via sigmoid-ish transform
    prob = 1 / (1 + np.exp(-(signal - 1.6)))
    y = (rng.random(n) < prob).astype(int).values
    return df, y


def main() -> None:
    X, y = make_data()

    num_cols = ["age", "income", "visits"]
    cat_cols = ["city", "device"]

    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", model),
        ]
    )

    # Cross-validated evaluation (leakage-safe)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_cv = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
    print("CV ROC-AUC mean:", round(float(auc_cv.mean()), 4), "std:", round(float(auc_cv.std()), 4))

    # Final held-out test evaluation
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1]
    print("Test ROC-AUC:", round(float(roc_auc_score(yte, proba)), 4))


if __name__ == "__main__":
    main()
