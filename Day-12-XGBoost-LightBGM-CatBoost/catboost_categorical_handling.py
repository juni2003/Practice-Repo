"""
CatBoost — Native Categorical Handling with Early Stopping

- Synthetic dataset with mixed numeric + categorical features
- Train/validation/test split
- Early stopping, feature importance
- Demonstrates passing categorical feature indices

Requires: catboost, pandas, scikit-learn, matplotlib
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def build_dataset(n: int = 4000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(50_000, 15_000, n),
        "city": rng.choice(["A", "B", "C", "D", "E"], n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "member": rng.choice(["yes", "no"], n),
    })
    # Binary target influenced by age/income/city/device + noise
    score = (
        (df["age"] > 40).astype(int) +
        (df["income"] > 60000).astype(int) +
        (df["city"].isin(["A", "B"])).astype(int) +
        (df["device"] == "desktop").astype(int) +
        (df["member"] == "yes").astype(int) +
        rng.integers(0, 2, n)
    )
    y = (score > 2).astype(int).values

    cat_idx = [2, 3, 4]
    return df, y, cat_idx


def main():
    df, y, cat_idx = build_dataset()
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=0.3, stratify=y, random_state=42)
    Xtr, Xval, ytr, yval = train_test_split(Xtr, ytr, test_size=0.25, stratify=ytr, random_state=42)

    train_pool = Pool(Xtr, ytr, cat_features=cat_idx)
    val_pool = Pool(Xval, yval, cat_features=cat_idx)
    test_pool = Pool(Xte, yte, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_state=42,
        verbose=False
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

    print("Best iteration:", model.get_best_iteration())
    y_pred = model.predict(test_pool).astype(int).reshape(-1)
    y_proba = model.predict_proba(test_pool)[:, 1]
    print("Accuracy:", accuracy_score(yte, y_pred))
    print("ROC-AUC:", roc_auc_score(yte, y_proba))

    # Feature importance
    imp = model.get_feature_importance(train_pool)
    order = np.argsort(imp)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh([df.columns[i] for i in order], imp[order], color="darkorange")
    plt.gca().invert_yaxis()
    plt.title("CatBoost Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
