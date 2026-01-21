"""
AdaBoost (Binary Classification) — From Scratch (Educational)

- Decision stumps as weak learners (single-threshold split per feature)
- SAMME (exponential) loss intuition: reweight samples; alpha_t ~ 0.5*ln((1-err)/err)
- Works for y in {0,1} but internally uses {-1,+1} for math

Note: For real projects, use sklearn.ensemble.AdaBoostClassifier.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


@dataclass
class Stump:
    feature: int
    threshold: float
    polarity: int  # +1 or -1
    alpha: float   # learner weight


def weighted_stump(X: np.ndarray, y_pm1: np.ndarray, w: np.ndarray) -> Tuple[Stump, np.ndarray, float]:
    """
    Train a decision stump by scanning thresholds per feature, minimizing weighted error.
    y_pm1: labels in {-1,+1}
    Returns stump, predictions (+/-1), and weighted error.
    """
    n, d = X.shape
    best_err = np.inf
    best_stump = None
    best_pred = None

    for j in range(d):
        # Sort by feature for efficient threshold checks
        idx = np.argsort(X[:, j])
        xj = X[idx, j]
        yj = y_pm1[idx]
        wj = w[idx]

        # Candidate thresholds: midpoints between unique sorted values
        uniq = np.unique(xj)
        if len(uniq) == 1:
            # No split possible
            continue
        thr_candidates = (uniq[:-1] + uniq[1:]) / 2.0

        for polarity in (+1, -1):
            # Prediction rule: pred = polarity * sign(xj - thr)
            # We'll compute efficiently by comparing to thresholds
            left_label = -1 * polarity
            right_label = +1 * polarity

            # Cumulative weights for left/right if we split at each threshold index
            # We'll scan threshold index t meaning: x <= thr is left, x > thr is right
            # Precompute misclassification for current polarity:
            # For left side (<= thr): predict left_label; error weight = sum_w(y != left_label)
            # For right side (> thr): predict right_label
            # We'll accumulate cumulative sums of weights where y != predicted.
            # Create indicator arrays for misclassification on left/right per position
            mis_left = (yj != left_label).astype(float) * wj
            mis_right = (yj != right_label).astype(float) * wj

            # cumulative sums
            csum_left = np.cumsum(mis_left)
            csum_right_rev = np.cumsum(mis_right[::-1])[::-1]

            # Map thresholds to indices: threshold between i and i+1 (0..n-2)
            # But we only consider unique value midpoints; need indices of last occurrence of each uniq value
            # Build mapping from uniq value position to last index in xj
            last_idx_for_val = np.searchsorted(xj, uniq, side="right") - 1  # positions of last occurrence
            # We'll evaluate split after all occurrences of uniq[k] are sent to left
            for k in range(len(uniq) - 1):
                i = last_idx_for_val[k]  # split after i
                err = csum_left[i] + (csum_right_rev[i + 1] if i + 1 < n else 0.0)
                if err < best_err:
                    thr = thr_candidates[k]
                    pred = np.where(X[:, j] <= thr, left_label, right_label)
                    best_err = err
                    best_stump = Stump(feature=j, threshold=thr, polarity=polarity, alpha=0.0)
                    best_pred = pred

    # Avoid division by zero; clip error to (1e-12, 1-1e-12)
    eps = 1e-12
    err_rate = np.clip(best_err, eps, 1 - eps)
    alpha = 0.5 * np.log((1 - err_rate) / err_rate)
    best_stump.alpha = float(alpha)
    return best_stump, best_pred, float(err_rate)


class AdaBoostScratch:
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.stumps: List[Stump] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        assert len(self.classes_) == 2, "Binary classification only"
        # Map to {-1, +1}
        y_pm1 = np.where(y == self.classes_[0], -1, +1).astype(float)

        n = X.shape[0]
        w = np.ones(n) / n
        self.stumps = []

        for t in range(self.n_estimators):
            stump, pred_pm1, err = weighted_stump(X, y_pm1, w)
            # Update sample weights: w_i <- w_i * exp(-alpha * y_i * h_t(x_i)), then normalize
            w *= np.exp(-stump.alpha * y_pm1 * pred_pm1)
            w /= w.sum()
            self.stumps.append(stump)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        F = np.zeros(X.shape[0])
        for s in self.stumps:
            left_label = -1 * s.polarity
            right_label = +1 * s.polarity
            pred = np.where(X[:, s.feature] <= s.threshold, left_label, right_label)
            F += s.alpha * pred
        return F

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.decision_function(X)
        labels_pm1 = np.sign(F)
        labels_pm1[labels_pm1 == 0] = 1  # tie-break
        # Map back to original class labels
        return np.where(labels_pm1 == -1, self.classes_[0], self.classes_[1])


def demo():
    X, y = make_classification(
        n_samples=1000, n_features=5, n_informative=2, n_redundant=0,
        class_sep=1.0, random_state=42
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    ada = AdaBoostScratch(n_estimators=100).fit(Xtr, ytr)
    ypred = ada.predict(Xte)
    print("AdaBoostScratch Accuracy:", accuracy_score(yte, ypred))
    print("\nReport:\n", classification_report(yte, ypred))


if __name__ == "__main__":
    demo()
  
