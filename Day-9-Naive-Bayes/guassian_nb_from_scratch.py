"""
Gaussian Naive Bayes — From Scratch (Educational)

Implements:
- Fit: per-class mean, variance per feature; class priors
- Predict: log posterior using Gaussian log-likelihood + log prior
- Demo on synthetic continuous data

Note: For production use sklearn.naive_bayes.GaussianNB.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class GaussianNBScratch:
    def __init__(self, var_smoothing: float = 1e-9):
        """
        var_smoothing adds a small value to variances to avoid division by zero.
        """
        self.var_smoothing = var_smoothing
        self.class_priors_: Dict[int, float] = {}
        self.means_: Dict[int, np.ndarray] = {}
        self.vars_: Dict[int, np.ndarray] = {}
        self.classes_: np.ndarray | None = None
        self.fitted_: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNBScratch":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]

        for c in self.classes_:
            Xc = X[y == c]
            self.class_priors_[c] = Xc.shape[0] / n_samples
            mean = Xc.mean(axis=0)
            var = Xc.var(axis=0) + self.var_smoothing
            self.means_[c] = mean
            self.vars_[c] = var

        self.fitted_ = True
        return self

    def _log_gaussian_likelihood(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        # Sum over features
        # log N(x; mean, var) = -0.5 * [log(2π var) + (x-mean)^2 / var]
        log_term = -0.5 * np.log(2.0 * np.pi * var)
        quad_term = -0.5 * ((x - mean) ** 2) / var
        return float(np.sum(log_term + quad_term))

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.fitted_, "Call fit() first"
        X = np.asarray(X, dtype=float)
        log_proba = np.zeros((X.shape[0], len(self.classes_)))
        for i, x in enumerate(X):
            for j, c in enumerate(self.classes_):
                lp = np.log(self.class_priors_[c])
                lp += self._log_gaussian_likelihood(x, self.means_[c], self.vars_[c])
                log_proba[i, j] = lp
        # Normalize for numerical stability (log-softmax)
        # subtract max log to avoid overflow
        maxlog = log_proba.max(axis=1, keepdims=True)
        log_proba = log_proba - maxlog
        # softmax in log-space is not needed for argmax, but return normalized for completeness
        # convert to probabilities
        proba = np.exp(log_proba)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return np.log(proba)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        log_proba = self.predict_log_proba(X)
        idx = np.argmax(log_proba, axis=1)
        return self.classes_[idx]


def demo():
    X, y = make_classification(
        n_samples=800,
        n_features=6,
        n_informative=6,
        n_redundant=0,
        random_state=42,
        class_sep=1.5,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    nb = GaussianNBScratch(var_smoothing=1e-9)
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    demo()
