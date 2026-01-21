"""
Scikit-learn Ensembles — Bagging, Random Forest, AdaBoost, Gradient Boosting

- Dataset: make_moons (classification)
- Compare against a single decision tree baseline
- Print accuracy and simple report
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)


def run():
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    # Single tree baseline
    tree = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree.fit(Xtr, ytr)
    ypred_tree = tree.predict(Xte)
    print("Single Tree Acc:", accuracy_score(yte, ypred_tree))

    # Bagging (with deep trees)
    bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=None, random_state=42),
        n_estimators=200, max_samples=0.8, bootstrap=True, random_state=42, n_jobs=-1
    ).fit(Xtr, ytr)
    ypred_bag = bag.predict(Xte)
    print("Bagging Acc    :", accuracy_score(yte, ypred_bag))

    # Random Forest (bagging + feature subsampling)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt", n_jobs=-1, random_state=42
    ).fit(Xtr, ytr)
    ypred_rf = rf.predict(Xte)
    print("RandomForest Acc:", accuracy_score(yte, ypred_rf))

    # AdaBoost (stumps)
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=300, learning_rate=0.1, random_state=42
    ).fit(Xtr, ytr)
    ypred_ada = ada.predict(Xte)
    print("AdaBoost Acc   :", accuracy_score(yte, ypred_ada))

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    ).fit(Xtr, ytr)
    ypred_gb = gb.predict(Xte)
    print("GradBoost Acc  :", accuracy_score(yte, ypred_gb))

    print("\nClassification report (Gradient Boosting):")
    print(classification_report(yte, ypred_gb))


if __name__ == "__main__":
    run()
