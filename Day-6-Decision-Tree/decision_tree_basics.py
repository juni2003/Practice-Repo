"""
Day 6 — Decision Tree Basics
-----------------------------

This file introduces sklearn's DecisionTreeClassifier with simple examples.  
Covers:
- Creating and training decision tree
- Making predictions
- Understanding key hyperparameters
- Comparing different splitting criteria (gini vs entropy)

Run:
    python Day-6-Decision-Trees-Random-Forests/decision_tree_basics. py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


def simple_tree_example():
    """
    Simple example using the Iris dataset.  
    Decision trees work well with multiclass classification.
    """
    print("="*60)
    print("Simple Decision Tree Example (Iris Dataset)")
    print("="*60)

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create a decision tree classifier
    # max_depth limits tree depth to prevent overfitting
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    print(f"Test accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))


def compare_splitting_criteria():
    """
    Compare Gini impurity vs Entropy as splitting criteria.
    Both work well, but Gini is faster.
    """
    print("\n" + "="*60)
    print("Comparing Gini vs Entropy")
    print("="*60)

    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=8,
        n_redundant=2, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Gini criterion (default)
    tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    tree_gini.fit(X_train, y_train)
    gini_accuracy = tree_gini.score(X_test, y_test)

    # Entropy criterion (information gain)
    tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)
    tree_entropy.fit(X_train, y_train)
    entropy_accuracy = tree_entropy.score(X_test, y_test)

    print(f"\nGini Impurity — Test Accuracy: {gini_accuracy:.4f}")
    print(f"Entropy (Info Gain) — Test Accuracy: {entropy_accuracy:.4f}")
    print("\nBoth criteria give similar results. Gini is slightly faster to compute.")


def hyperparameter_exploration():
    """
    Explore key hyperparameters:
    - max_depth: Maximum depth of the tree
    - min_samples_split: Minimum samples required to split a node
    - min_samples_leaf: Minimum samples required in  leaf node
    """
    print("\n" + "="*60)
    print("Hyperparameter Exploration")
    print("="*60)

    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=10,
        n_redundant=5, random_state=42
    )

    print("\nTesting different max_depth values:")
    for depth in [None, 3, 5, 10]:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
        print(f"  max_depth={str(depth):5s} → CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\nTesting different min_samples_split values:")
    for min_split in [2, 10, 50, 100]:
        tree = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
        scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
        print(f"  min_samples_split={min_split:3d} → CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    print("\nKey Insight:")
    print("- max_depth=None → Tree grows until pure (overfitting risk)")
    print("- Limiting max_depth or increasing min_samples_split prevents overfitting")


def tree_structure_info():
    """
    Extract and display tree structure information.
    """
    print("\n" + "="*60)
    print("Tree Structure Information")
    print("="*60)

    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)

    print(f"\nTree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")
    print(f"Number of nodes: {tree.tree_.node_count}")

    # Feature importances (how much each feature contributes to splits)
    importances = tree.feature_importances_
    print("\nFeature Importances:")
    for i, importance in enumerate(importances):
        print(f"  Feature {i}: {importance:.4f}")


def main():
    """
    Run all decision tree basics examples.
    """
    simple_tree_example()
    compare_splitting_criteria()
    hyperparameter_exploration()
    tree_structure_info()

    print("\n" + "="*60)
    print("Decision Tree Basics — Complete!")
    print("="*60)
    print("\nNext: Run entropy_information_gain.py to understand the math behind splits.")


if __name__ == "__main__":
    main()
