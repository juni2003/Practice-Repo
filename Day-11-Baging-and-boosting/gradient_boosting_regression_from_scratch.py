"""
Gradient Boosting (Regression) — From Scratch (Squared Error)

- Initialize with mean target
- Iteratively fit shallow trees to residuals (negative gradients)
- Update: F_{m}(x) = F_{m-1}(x) + η * h_m(x)
- Tracks train/test MSE vs n_estimators

Note: For production, use sklearn.ensemble.GradientBoostingRegressor or HistGradientBoostingRegressor.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


@dataclass
class GBRegScratch:
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    init_: float = 0.0
    trees_: Optional[List[DecisionTreeRegressor]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.init_ = float(np.mean(y))
        self.trees_ = []
        F = np.full_like(y, self.init_, dtype=float)

        for m in range(self.n_estimators):
            residual = y - F  # negative gradient of squared error
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=123 + m)
            tree.fit(X, residual)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F


def demo():
    X, y = make_regression(n_samples=1200, n_features=10, noise=20.0, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

    model = GBRegScratch(n_estimators=200, learning_rate=0.1, max_depth=3).fit(Xtr, ytr)

    # Track MSE vs n_estimators
    mses_tr, mses_te = [], []
    Ftr = np.full(Xtr.shape[0], model.init_)
    Fte = np.full(Xte.shape[0], model.init_)
    for tree in model.trees_:
        Ftr += model.learning_rate * tree.predict(Xtr)
        Fte += model.learning_rate * tree.predict(Xte)
        mses_tr.append(mean_squared_error(ytr, Ftr))
        mses_te.append(mean_squared_error(yte, Fte))

    print("Final Train/Test MSE:", mses_tr[-1], mses_te[-1])

    # Plot curves
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        plt.plot(range(1, len(mses_tr) + 1), mses_tr, label="Train MSE")
        plt.plot(range(1, len(mses_te) + 1), mses_te, label="Test MSE")
        plt.xlabel("n_estimators")
        plt.ylabel("MSE")
        plt.title("Gradient Boosting Regression — MSE vs n_estimators")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    demo()
