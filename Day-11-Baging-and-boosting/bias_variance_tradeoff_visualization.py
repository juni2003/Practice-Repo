"""
Bias–Variance Tradeoff Visualization with Bagging

- Vary n_estimators and base tree depth
- Plot train vs test accuracy to observe bias/variance behavior
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


def run(depths=(1, 3, None), n_estimators_list=(1, 5, 10, 25, 50, 100, 200)):
    X, y = make_moons(n_samples=1200, noise=0.3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    plt.figure(figsize=(12, 8))
    for idx, depth in enumerate(depths, 1):
        train_acc, test_acc = [], []
        for n in n_estimators_list:
            base = DecisionTreeClassifier(max_depth=depth, random_state=42)
            bag = BaggingClassifier(
                estimator=base, n_estimators=n, max_samples=0.8, bootstrap=True,
                random_state=42, n_jobs=-1
            )
            bag.fit(Xtr, ytr)
            train_acc.append(accuracy_score(ytr, bag.predict(Xtr)))
            test_acc.append(accuracy_score(yte, bag.predict(Xte)))

        plt.subplot(2, 2, idx)
        plt.plot(n_estimators_list, train_acc, "o--", label="Train")
        plt.plot(n_estimators_list, test_acc, "o-", label="Test")
        title = f"Base Tree Depth = {depth if depth is not None else 'Full'}"
        plt.title(title)
        plt.xlabel("n_estimators")
        plt.ylabel("Accuracy")
        plt.ylim(0.5, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
