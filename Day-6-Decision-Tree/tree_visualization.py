"""
Day 6 — Tree Visualization
--------------------------

This file demonstrates:
- Visualizing decision trees using sklearn's plot_tree
- Visualizing decision boundaries in 2D
- Exporting trees to text format

Run:
    python Day-6-Decision-Trees-Random-Forests/tree_visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split


def visualize_tree_structure():
    """
    Visualize the structure of a decision tree using sklearn's plot_tree.
    """
    print("="*60)
    print("Visualizing Tree Structure")
    print("="*60)
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=iris. feature_names,
        class_names=iris.target_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Decision Tree Structure (Iris Dataset)", fontsize=16)
    plt.tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/tree_structure. png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nSaved tree structure visualization: {output_path}")
    plt.close()
    
    print("\nTree interpretation:")
    print("- Each box is a node")
    print("- Top line: split condition (e.g., 'petal length <= 2.45')")
    print("- 'gini': impurity measure")
    print("- 'samples': number of samples at this node")
    print("- 'value': number of samples per class")
    print("- 'class': majority class at this node")
    print("- Color: intensity represents class purity")


def visualize_decision_boundary_2d():
    """
    Visualize decision boundary for a 2D dataset.
    Shows how the tree partitions the feature space.
    """
    print("\n" + "="*60)
    print("Visualizing Decision Boundary (2D)")
    print("="*60)
    
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree.fit(X, y)
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1]. max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx. shape)
    
    plt. figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt. scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k', cmap='RdYlBu')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt. title('Decision Tree Decision Boundary (max_depth=4)', fontsize=14)
    plt.colorbar(label='Class')
    plt.tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/decision_boundary. png"
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved decision boundary visualization: {output_path}")
    plt.close()
    
    print("\nObservation:")
    print("- Decision tree creates axis-aligned rectangular regions")
    print("- Each region corresponds to a leaf node")
    print("- Boundaries are perpendicular to feature axes")


def compare_tree_depths():
    """
    Compare decision boundaries for trees of different depths.
    Shows how depth affects complexity and overfitting.
    """
    print("\n" + "="*60)
    print("Comparing Tree Depths")
    print("="*60)
    
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    depths = [2, 4, 8, None]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, depth in enumerate(depths):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X, y)
        
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np. meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[idx].contourf(xx, yy, Z, alpha=0. 3, cmap='RdYlBu')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', cmap='RdYlBu')
        
        depth_str = str(depth) if depth is not None else "Unlimited"
        axes[idx].set_title(f'max_depth={depth_str} (leaves: {tree.get_n_leaves()})', fontsize=12)
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
    
    plt.suptitle('Decision Boundaries for Different Tree Depths', fontsize=16)
    plt. tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/tree_depths_comparison.png"
    plt. savefig(output_path, dpi=100)
    print(f"\nSaved tree depth comparison: {output_path}")
    plt.close()
    
    print("\nKey insight:")
    print("- Shallow trees (depth=2): Underfit, simple boundaries")
    print("- Medium trees (depth=4): Good balance")
    print("- Deep trees (depth=8+): Overfit, complex boundaries that memorize training data")


def export_tree_text():
    """
    Export tree structure as text rules.
    """
    print("\n" + "="*60)
    print("Exporting Tree as Text Rules")
    print("="*60)
    
    from sklearn.tree import export_text
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    tree_rules = export_text(tree, feature_names=iris.feature_names)
    
    print("\nTree as text rules:")
    print(tree_rules)
    
    output_file = "Day-6-Decision-Trees-Random-Forests/tree_rules.txt"
    with open(output_file, 'w') as f:
        f.write(tree_rules)
    
    print(f"\nSaved tree rules to: {output_file}")


def main():
    """
    Run all tree visualization examples.
    """
    visualize_tree_structure()
    visualize_decision_boundary_2d()
    compare_tree_depths()
    export_tree_text()
    
    print("\n" + "="*60)
    print("Tree Visualization — Complete!")
    print("="*60)
    print("\nNext: Run tree_pruning_example.py to learn about preventing overfitting.")


if __name__ == "__main__":
    main()
