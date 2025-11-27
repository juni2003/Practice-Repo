"""
Day 6 — Tree Pruning
--------------------

This file demonstrates:
- Pre-pruning (early stopping with hyperparameters)
- Post-pruning (cost-complexity pruning)
- Comparing pruned vs unpruned trees

Run:
    python Day-6-Decision-Trees-Random-Forests/tree_pruning_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


def pre_pruning_example():
    """
    Pre-pruning uses hyperparameters to stop tree growth early.
    
    Key parameters:
    - max_depth: Maximum depth
    - min_samples_split: Minimum samples to split a node
    - min_samples_leaf: Minimum samples in a leaf
    - max_leaf_nodes: Maximum number of leaves
    """
    print("="*60)
    print("Pre-Pruning (Early Stopping)")
    print("="*60)
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Unpruned tree (grows until pure)
    tree_unpruned = DecisionTreeClassifier(random_state=42)
    tree_unpruned.fit(X_train, y_train)
    
    train_score_unpruned = tree_unpruned.score(X_train, y_train)
    test_score_unpruned = tree_unpruned.score(X_test, y_test)
    
    print("\nUnpruned Tree:")
    print(f"  Depth: {tree_unpruned. get_depth()}")
    print(f"  Leaves: {tree_unpruned.get_n_leaves()}")
    print(f"  Train accuracy: {train_score_unpruned:.4f}")
    print(f"  Test accuracy:  {test_score_unpruned:.4f}")
    print(f"  Overfitting gap: {train_score_unpruned - test_score_unpruned:.4f}")
    
    # Pre-pruned tree (max_depth)
    tree_pruned = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
    tree_pruned.fit(X_train, y_train)
    
    train_score_pruned = tree_pruned. score(X_train, y_train)
    test_score_pruned = tree_pruned. score(X_test, y_test)
    
    print("\nPre-Pruned Tree (max_depth=5, min_samples_split=20):")
    print(f"  Depth: {tree_pruned.get_depth()}")
    print(f"  Leaves: {tree_pruned.get_n_leaves()}")
    print(f"  Train accuracy: {train_score_pruned:.4f}")
    print(f"  Test accuracy:  {test_score_pruned:.4f}")
    print(f"  Overfitting gap: {train_score_pruned - test_score_pruned:.4f}")
    
    print("\n📊 Conclusion:")
    print("  Pre-pruning reduces overfitting by limiting tree complexity.")
    print("  Test accuracy improves even though training accuracy decreases.")


def post_pruning_cost_complexity():
    """
    Post-pruning (cost-complexity pruning) grows a full tree then removes branches.
    
    Uses the ccp_alpha parameter:
    - ccp_alpha = 0: No pruning (full tree)
    - Higher ccp_alpha: More aggressive pruning
    """
    print("\n" + "="*60)
    print("Post-Pruning (Cost-Complexity Pruning)")
    print("="*60)
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # First, fit a full tree to get cost-complexity path
    tree = DecisionTreeClassifier(random_state=42)
    path = tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities
    
    print(f"\nFound {len(ccp_alphas)} cost-complexity alpha values")
    print(f"Alpha range: {ccp_alphas. min():.6f} to {ccp_alphas.max():.6f}")
    
    # Train trees with different ccp_alpha values
    train_scores = []
    test_scores = []
    
    for ccp_alpha in ccp_alphas[::10]:
        tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        train_scores.append(tree.score(X_train, y_train))
        test_scores.append(tree.score(X_test, y_test))
    
    # Find best alpha
    best_idx = np. argmax(test_scores)
    best_alpha = ccp_alphas[::10][best_idx]
    
    print(f"\nBest ccp_alpha: {best_alpha:.6f}")
    print(f"Best test accuracy: {test_scores[best_idx]:.4f}")
    
    # Visualize pruning effect
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(ccp_alphas[::10], train_scores, marker='o', label='Train', alpha=0.7)
    plt.plot(ccp_alphas[::10], test_scores, marker='s', label='Test', alpha=0.7)
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best alpha={best_alpha:.6f}')
    plt.xlabel('ccp_alpha (pruning strength)')
    plt.ylabel('Accuracy')
    plt.title('Cost-Complexity Pruning: Accuracy vs Alpha')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Number of nodes vs alpha
    plt.subplot(1, 2, 2)
    node_counts = []
    for ccp_alpha in ccp_alphas[::10]:
        tree = DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
        tree.fit(X_train, y_train)
        node_counts.append(tree.tree_.node_count)
    
    plt.plot(ccp_alphas[::10], node_counts, marker='o', color='green')
    plt.axvline(best_alpha, color='red', linestyle='--', label=f'Best alpha={best_alpha:.6f}')
    plt.xlabel('ccp_alpha (pruning strength)')
    plt.ylabel('Number of Nodes')
    plt. title('Tree Complexity vs Alpha')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = "Day-6-Decision-Trees-Random-Forests/cost_complexity_pruning.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved cost-complexity pruning plot: {output_path}")
    plt.close()
    
    print("\n📊 Conclusion:")
    print("  - As ccp_alpha increases, tree becomes simpler (fewer nodes)")
    print("  - Test accuracy peaks at optimal alpha, then decreases (underfitting)")
    print("  - Cost-complexity pruning automatically finds the right balance")


def compare_pruning_strategies():
    """
    Compare pre-pruning vs post-pruning. 
    """
    print("\n" + "="*60)
    print("Comparing Pruning Strategies")
    print("="*60)
    
    X, y = make_classification(
        n_samples=800,
        n_features=20,
        n_informative=12,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # No pruning
    tree_none = DecisionTreeClassifier(random_state=42)
    cv_none = cross_val_score(tree_none, X_train, y_train, cv=5). mean()
    
    # Pre-pruning
    tree_pre = DecisionTreeClassifier(max_depth=7, min_samples_split=15, random_state=42)
    cv_pre = cross_val_score(tree_pre, X_train, y_train, cv=5).mean()
    
    # Post-pruning
    tree_post = DecisionTreeClassifier(ccp_alpha=0.01, random_state=42)
    cv_post = cross_val_score(tree_post, X_train, y_train, cv=5).mean()
    
    print("\nCross-Validation Results (5-fold):")
    print(f"  No pruning:     {cv_none:.4f}")
    print(f"  Pre-pruning:    {cv_pre:.4f}")
    print(f"  Post-pruning:   {cv_post:.4f}")
    
    print("\n📊 Recommendation:")
    print("  - Pre-pruning: Faster, good for quick experiments")
    print("  - Post-pruning: More principled, finds optimal complexity automatically")
    print("  - In practice: Try both and use cross-validation to compare")


def main():
    """
    Run all tree pruning examples.
    """
    pre_pruning_example()
    post_pruning_cost_complexity()
    compare_pruning_strategies()
    
    print("\n" + "="*60)
    print("Tree Pruning — Complete!")
    print("="*60)
    print("\nNext: Run random_forest_ensemble. py to learn about ensemble methods.")


if __name__ == "__main__":
    main()
