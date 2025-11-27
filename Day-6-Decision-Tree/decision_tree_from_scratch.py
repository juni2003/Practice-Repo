"""
Day 6 — Decision Tree From Scratch
-----------------------------------

This file implements a decision tree classifier from scratch using:
- Entropy and Information Gain
- Recursive tree building
- Prediction logic

This is educational code to understand how decision trees work internally.
For production, use sklearn. tree.DecisionTreeClassifier. 

Run:
    python Day-6-Decision-Trees-Random-Forests/decision_tree_from_scratch.py
"""

import numpy as np
from collections import Counter


class Node:
    """
    Represents a node in the decision tree.
    
    A node can be either:
    - Internal node: Has a feature to split on and threshold
    - Leaf node: Has a class label (value)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None


class DecisionTreeFromScratch:
    """
    Decision Tree Classifier built from scratch.
    
    Uses entropy and information gain to find best splits.
    Supports binary and multiclass classification.
    """
    
    def __init__(self, max_depth=10, min_samples_split=2):
        """
        Initialize the decision tree. 
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y):
        """
        Build the decision tree from training data.
        
        Args:
            X: Training features (numpy array)
            y: Training labels (numpy array)
        """
        self.root = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree. 
        
        Args:
            X: Feature matrix
            y: Labels
            depth: Current depth in the tree
        
        Returns:
            Node object (either internal node or leaf)
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on.
        
        Tries all features and all possible thresholds (midpoints between unique values).
        Returns the split with maximum information gain.
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            best_feature: Index of best feature
            best_threshold: Best threshold value
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        parent_entropy = self._entropy(y)
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Try every feature
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Try every threshold (midpoint between consecutive unique values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate information gain
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                n_left = len(left_y)
                n_right = len(right_y)
                
                weighted_entropy = (
                    (n_left / n_samples) * self._entropy(left_y) +
                    (n_right / n_samples) * self._entropy(right_y)
                )
                
                gain = parent_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _entropy(self, y):
        """
        Calculate entropy of labels.
        
        Entropy = -Σ p_i * log₂(p_i)
        
        Args:
            y: Array of labels
        
        Returns:
            Entropy value
        """
        counts = np.bincount(y)
        probabilities = counts[counts > 0] / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def _most_common_label(self, y):
        """
        Return the most common label in y.
        Used for leaf node prediction.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: Feature matrix (numpy array)
        
        Returns:
            Array of predicted labels
        """
        return np. array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to make a prediction for a single sample.
        
        Args:
            x: Single sample (feature vector)
            node: Current node
        
        Returns:
            Predicted class label
        """
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node. left)
        else:
            return self._traverse_tree(x, node.right)
    
    def print_tree(self, node=None, depth=0):
        """
        Print the tree structure (for debugging/visualization).
        
        Args:
            node: Current node (default: root)
            depth: Current depth (for indentation)
        """
        if node is None:
            node = self.root
        
        indent = "  " * depth
        
        if node.is_leaf():
            print(f"{indent}Leaf: class={node.value}")
        else:
            print(f"{indent}Node: feature_{node.feature} <= {node.threshold:. 4f}")
            print(f"{indent}Left:")
            self.print_tree(node.left, depth + 1)
            print(f"{indent}Right:")
            self.print_tree(node.right, depth + 1)


def test_on_iris():
    """
    Test the decision tree on the Iris dataset.
    """
    print("="*60)
    print("Testing Decision Tree From Scratch on Iris Dataset")
    print("="*60)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    tree = DecisionTreeFromScratch(max_depth=5, min_samples_split=2)
    tree.fit(X_train, y_train)
    
    y_pred = tree. predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nTree Structure (first few levels):")
    tree.print_tree()


def compare_with_sklearn():
    """
    Compare our implementation with sklearn's DecisionTreeClassifier.
    """
    print("\n" + "="*60)
    print("Comparing with sklearn DecisionTreeClassifier")
    print("="*60)
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=7,
        n_redundant=3, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Our implementation
    our_tree = DecisionTreeFromScratch(max_depth=5, min_samples_split=5)
    our_tree.fit(X_train, y_train)
    our_pred = our_tree.predict(X_test)
    our_accuracy = accuracy_score(y_test, our_pred)
    
    # Sklearn implementation
    sklearn_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
    sklearn_tree.fit(X_train, y_train)
    sklearn_pred = sklearn_tree. predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    
    print(f"\nOur implementation accuracy:    {our_accuracy:.4f}")
    print(f"Sklearn implementation accuracy: {sklearn_accuracy:.4f}")
    print("\nNote: Small differences are expected due to tie-breaking and implementation details.")


def main():
    """
    Run all from-scratch decision tree examples.
    """
    test_on_iris()
    compare_with_sklearn()
    
    print("\n" + "="*60)
    print("Decision Tree From Scratch — Complete!")
    print("="*60)
    print("\nNext: Run tree_visualization.py to visualize decision trees.")


if __name__ == "__main__":
    main()
