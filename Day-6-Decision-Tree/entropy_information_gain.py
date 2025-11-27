"""
Day 6 — Entropy and Information Gain
-------------------------------------

This file demonstrates:
- Manual calculation of entropy
- Manual calculation of information gain
- Visualizing entropy for binary classification
- Finding the best split using information gain

Run:
    python Day-6-Decision-Trees-Random-Forests/entropy_information_gain.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def calculate_entropy(labels):
    """
    Calculate entropy of a dataset.
    
    Entropy measures impurity/disorder:
    - Entropy = 0 → All samples belong to one class (pure)
    - Entropy = 1 → Equal distribution of classes (maximum impurity)
    
    Formula: Entropy(S) = -Σ p_i * log₂(p_i)
    
    Args:
        labels: List or array of class labels
    
    Returns:
        Entropy value (float)
    """
    if len(labels) == 0:
        return 0.0
    
    counts = Counter(labels)
    total = len(labels)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy


def calculate_information_gain(parent_labels, left_labels, right_labels):
    """
    Calculate information gain from a split.
    
    Information Gain = Entropy(parent) - Weighted Average Entropy(children)
    
    Higher information gain = better split
    
    Args:
        parent_labels: Labels before split
        left_labels: Labels in left child
        right_labels: Labels in right child
    
    Returns:
        Information gain (float)
    """
    parent_entropy = calculate_entropy(parent_labels)
    
    n_parent = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    
    if n_parent == 0:
        return 0.0
    
    weighted_child_entropy = (
        (n_left / n_parent) * calculate_entropy(left_labels) +
        (n_right / n_parent) * calculate_entropy(right_labels)
    )
    
    information_gain = parent_entropy - weighted_child_entropy
    
    return information_gain


def entropy_examples():
    """
    Show entropy calculation examples with different class distributions.
    """
    print("="*60)
    print("Entropy Calculation Examples")
    print("="*60)
    
    examples = [
        ("Pure (all same class)", [1, 1, 1, 1, 1]),
        ("Maximum impurity (50-50)", [0, 0, 1, 1]),
        ("Mostly one class (80-20)", [0, 0, 0, 0, 1]),
        ("Three classes (equal)", [0, 1, 2, 0, 1, 2]),
        ("Three classes (unequal)", [0, 0, 0, 1, 1, 2]),
    ]
    
    for name, labels in examples:
        entropy = calculate_entropy(labels)
        counts = Counter(labels)
        print(f"\n{name}")
        print(f"  Labels: {labels}")
        print(f"  Class distribution: {dict(counts)}")
        print(f"  Entropy: {entropy:.4f}")


def information_gain_example():
    """
    Demonstrate information gain calculation with a concrete example.
    """
    print("\n" + "="*60)
    print("Information Gain Example")
    print("="*60)
    
    # Example: Should we approve a loan?
    # Features: Income, Age
    # Target: Loan approved (1) or rejected (0)
    
    print("\nDataset: Loan Approval")
    print("Features: Age, Income")
    print("Target: Approved (1) or Rejected (0)")
    
    parent_labels = [1, 1, 0, 0, 1, 0, 1, 0]
    parent_entropy = calculate_entropy(parent_labels)
    
    print(f"\nParent node labels: {parent_labels}")
    print(f"Parent entropy: {parent_entropy:.4f}")
    
    # Split 1: Age < 30
    print("\n--- Split 1: Age < 30 ---")
    left_split1 = [1, 1, 0]
    right_split1 = [0, 1, 0, 1, 0]
    
    ig1 = calculate_information_gain(parent_labels, left_split1, right_split1)
    print(f"Left child (Age < 30): {left_split1} → Entropy: {calculate_entropy(left_split1):.4f}")
    print(f"Right child (Age >= 30): {right_split1} → Entropy: {calculate_entropy(right_split1):. 4f}")
    print(f"Information Gain: {ig1:.4f}")
    
    # Split 2: Income > 50k
    print("\n--- Split 2: Income > 50k ---")
    left_split2 = [1, 1, 1, 1]
    right_split2 = [0, 0, 0, 0]
    
    ig2 = calculate_information_gain(parent_labels, left_split2, right_split2)
    print(f"Left child (Income > 50k): {left_split2} → Entropy: {calculate_entropy(left_split2):. 4f}")
    print(f"Right child (Income <= 50k): {right_split2} → Entropy: {calculate_entropy(right_split2):.4f}")
    print(f"Information Gain: {ig2:. 4f}")
    
    print("\n📊 Conclusion:")
    print(f"  Split by Income gives higher information gain ({ig2:.4f} > {ig1:.4f})")
    print(f"  → We should split on Income first!")


def visualize_entropy():
    """
    Visualize entropy as a function of class probability (binary classification).
    """
    print("\n" + "="*60)
    print("Visualizing Entropy")
    print("="*60)
    
    p_values = np.linspace(0. 01, 0.99, 100)
    entropy_values = [-p * np.log2(p) - (1-p) * np. log2(1-p) for p in p_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(p_values, entropy_values, linewidth=2, color='steelblue')
    plt. axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Maximum entropy (p=0. 5)')
    plt.xlabel('Probability of Class 1 (p)', fontsize=12)
    plt.ylabel('Entropy', fontsize=12)
    plt.title('Entropy vs Class Probability (Binary Classification)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/entropy_visualization.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved plot: {output_path}")
    plt.close()
    
    print("\nKey insight:")
    print("  - Entropy is maximum (1. 0) when classes are equally distributed (p=0.5)")
    print("  - Entropy is minimum (0.0) when all samples belong to one class (p=0 or p=1)")


def find_best_split_example():
    """
    Given a small dataset, find the best feature to split on.
    """
    print("\n" + "="*60)
    print("Finding the Best Split")
    print("="*60)
    
    data = {
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                    'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                       'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                    'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
                'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    }
    
    df = pd.DataFrame(data)
    print("\nDataset:")
    print(df)
    
    target = df['Play'].map({'Yes': 1, 'No': 0}). values
    parent_entropy = calculate_entropy(target)
    
    print(f"\nParent entropy: {parent_entropy:.4f}")
    
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    information_gains = {}
    
    for feature in features:
        feature_values = df[feature].unique()
        
        weighted_entropy = 0.0
        for value in feature_values:
            subset_target = target[df[feature] == value]
            weight = len(subset_target) / len(target)
            weighted_entropy += weight * calculate_entropy(subset_target)
        
        ig = parent_entropy - weighted_entropy
        information_gains[feature] = ig
    
    print("\nInformation Gain for each feature:")
    for feature, ig in sorted(information_gains.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature:15s}: {ig:.4f}")
    
    best_feature = max(information_gains, key=information_gains.get)
    print(f"\n🏆 Best feature to split on: {best_feature}")


def main():
    """
    Run all entropy and information gain examples. 
    """
    entropy_examples()
    information_gain_example()
    visualize_entropy()
    find_best_split_example()
    
    print("\n" + "="*60)
    print("Entropy & Information Gain — Complete!")
    print("="*60)
    print("\nNext: Run decision_tree_from_scratch.py to build a tree using these concepts.")


if __name__ == "__main__":
    main()
