"""
Bagging vs Pasting — Comparison with OOB Score

- Bagging: bootstrap sampling (with replacement), supports OOB score
- Pasting: sampling without replacement
- Both reduce variance by averaging; bagging typically stronger due to bootstrap diversity
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


def main(plot_boundary: bool = True):
    X, y = make_moons(n_samples=800, noise=0.3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    base = DecisionTreeClassifier(max_depth=None, random_state=42)

    bag = BaggingClassifier(
        estimator=base, n_estimators=200, max_samples=0.8, bootstrap=True,
        oob_score=True, random_state=42, n_jobs=-1
    )
    bag.fit(Xtr, ytr)
    ypred_bag = bag.predict(Xte)
    acc_bag = accuracy_score(yte, ypred_bag)

    paste = BaggingClassifier(
        estimator=base, n_estimators=200, max_samples=0.8, bootstrap=False,
        oob_score=False, random_state=42, n_jobs=-1
    )
    paste.fit(Xtr, ytr)
    ypred_paste = paste.predict(Xte)
    acc_paste = accuracy_score(yte, ypred_paste)

    print(f"Bagging:  OOB={bag.oob_score_:.4f} | Test Acc={acc_bag:.4f}")
    print(f"Pasting:  OOB=  N/A         | Test Acc={acc_paste:.4f}")

    if plot_boundary:
        # Decision boundaries
        models = [("Bagging", bag), ("Pasting", paste)]
        h = 0.02
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        plt.figure(figsize=(10, 4))
        for i, (name, clf) in enumerate(models, 1):
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            plt.subplot(1, 2, i)
            plt.contourf(xx, yy, Z, cmap="RdYlBu", alpha=0.35)
            plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="RdYlBu", s=15, edgecolor="k", alpha=0.8)
            plt.title(f"{name} — Test Acc={accuracy_score(yte, clf.predict(Xte)):.3f}")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(plot_boundary=True)
