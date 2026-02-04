"""
Cross-Validation Strategies — Usage & Pitfalls

- StratifiedKFold: preserves class distribution (classification default)
- GroupKFold: keeps grouped samples together (no leakage across groups)
- TimeSeriesSplit: walk-forward splits for temporal data

This script shows simple, runnable examples and prints split indices.
"""

from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold, TimeSeriesSplit, train_test_split
from sklearn.datasets import make_classification


def stratified_example():
    X, y = make_classification(n_samples=20, n_features=5, n_informative=3, weights=[0.7, 0.3], random_state=0)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    print("StratifiedKFold splits (class balance preserved):")
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        print(f"  Fold {i}: train={len(tr)}, test={len(te)}, y_test_dist={np.bincount(y[te])}")


def group_example():
    rng = np.random.default_rng(42)
    n_groups = 5
    group_sizes = [10, 8, 12, 6, 9]
    groups = np.concatenate([np.full(s, i) for i, s in enumerate(group_sizes)])
    X = rng.normal(size=(sum(group_sizes), 4))
    y = rng.integers(0, 2, size=sum(group_sizes))

    cv = GroupKFold(n_splits=5)
    print("\nGroupKFold splits (no group leakage):")
    for i, (tr, te) in enumerate(cv.split(X, y, groups=groups), 1):
        unique_train_groups = np.unique(groups[tr])
        unique_test_groups = np.unique(groups[te])
        leak = len(set(unique_train_groups).intersection(set(unique_test_groups))) > 0
        print(f"  Fold {i}: train_groups={unique_train_groups}, test_groups={unique_test_groups}, leakage={leak}")


def timeseries_example():
    n = 30
    X = np.arange(n).reshape(-1, 1)
    y = (X[:, 0] % 2).astype(int)
    cv = TimeSeriesSplit(n_splits=4)
    print("\nTimeSeriesSplit (walk-forward; no shuffling):")
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        print(f"  Fold {i}: train_idx=[{tr[0]}..{tr[-1]}], test_idx=[{te[0]}..{te[-1]}]")


def main():
    stratified_example()
    group_example()
    timeseries_example()


if __name__ == "__main__":
    main()
