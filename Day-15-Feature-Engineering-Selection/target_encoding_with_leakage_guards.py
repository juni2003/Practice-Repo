"""
Target Encoding with Leakage Guards (Out-of-Fold Encoding)

Target encoding replaces a categorical value with a statistic of the target
(e.g., mean target for that category). Powerful for high-cardinality categories.

But if you compute category -> mean(target) on the full dataset BEFORE splitting,
you leak target information into validation/test.

This script implements out-of-fold (OOF) target encoding:
- For each fold, compute encoding using only the fold's training indices.
- Apply to the fold's validation indices.
- Then train a model on OOF-encoded training data, evaluate on test.

We also show a "leaky" version for comparison.

Requirements: numpy, pandas, scikit-learn
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def make_high_card_data(n: int = 6000, n_categories: int = 400, seed: int = 42):
    rng = np.random.default_rng(seed)
    # One high-cardinality column + a numeric feature
    cat = rng.integers(0, n_categories, size=n).astype(str)
    x_num = rng.normal(size=n)

    # Give some categories higher propensity
    cat_effect = rng.normal(0, 0.7, size=n_categories)
    logits = 0.8 * x_num + cat_effect[cat.astype(int)]
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({"cat_id": cat, "x_num": x_num})
    return df, y


def target_encode_fit(series: pd.Series, y: np.ndarray, smoothing: float = 10.0) -> dict:
    """
    Smoothing reduces overfitting for rare categories.

    enc(cat) = (sum_y + smoothing * global_mean) / (count + smoothing)
    """
    global_mean = float(np.mean(y))
    stats = pd.DataFrame({"cat": series.values, "y": y}).groupby("cat")["y"].agg(["sum", "count"])
    enc = (stats["sum"] + smoothing * global_mean) / (stats["count"] + smoothing)
    mapping = enc.to_dict()
    mapping["__global__"] = global_mean
    return mapping


def target_encode_apply(series: pd.Series, mapping: dict) -> np.ndarray:
    global_mean = mapping.get("__global__", 0.5)
    return series.map(mapping).fillna(global_mean).astype(float).values


def oof_target_encode(X: pd.DataFrame, y: np.ndarray, col: str, n_splits: int = 5, seed: int = 42) -> tuple[np.ndarray, list[dict]]:
    """
    Returns:
      oof_encoded: shape (n_samples,)
      fold_mappings: mapping used per fold (mainly for learning / debugging)
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=float)
    mappings: list[dict] = []

    for tr_idx, val_idx in cv.split(X, y):
        mapping = target_encode_fit(X.iloc[tr_idx][col], y[tr_idx], smoothing=10.0)
        oof[val_idx] = target_encode_apply(X.iloc[val_idx][col], mapping)
        mappings.append(mapping)

    return oof, mappings


def main() -> None:
    X, y = make_high_card_data()

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # --- Leaky encoding (WRONG): fit on full training+validation as if you knew y for all rows ---
    leaky_map = target_encode_fit(Xtr["cat_id"], ytr, smoothing=10.0)
    Xtr_leaky = Xtr.copy()
    Xte_leaky = Xte.copy()
    Xtr_leaky["cat_te"] = target_encode_apply(Xtr_leaky["cat_id"], leaky_map)
    Xte_leaky["cat_te"] = target_encode_apply(Xte_leaky["cat_id"], leaky_map)

    # --- OOF encoding (RIGHT): OOF for training, then fit mapping on full train and apply to test ---
    te_oof, _ = oof_target_encode(Xtr, ytr, col="cat_id", n_splits=5, seed=42)
    Xtr_oof = Xtr.copy()
    Xtr_oof["cat_te"] = te_oof

    final_map = target_encode_fit(Xtr["cat_id"], ytr, smoothing=10.0)
    Xte_oof = Xte.copy()
    Xte_oof["cat_te"] = target_encode_apply(Xte_oof["cat_id"], final_map)

    # Train a simple model
    feats = ["x_num", "cat_te"]
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)

    # Evaluate leaky
    clf.fit(Xtr_leaky[feats], ytr)
    auc_leaky = roc_auc_score(yte, clf.predict_proba(Xte_leaky[feats])[:, 1])

    # Evaluate OOF-safe
    clf.fit(Xtr_oof[feats], ytr)
    auc_oof = roc_auc_score(yte, clf.predict_proba(Xte_oof[feats])[:, 1])

    print("AUC with LEAKY target encoding (optimistic):", round(float(auc_leaky), 4))
    print("AUC with OOF-safe target encoding:", round(float(auc_oof), 4))
    print("\nRule: target encoding must be computed inside CV folds, never using targets from the validation fold.")


if __name__ == "__main__":
    main()
