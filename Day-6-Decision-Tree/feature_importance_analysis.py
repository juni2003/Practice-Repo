"""
Day 6 — Feature Importance Analysis
------------------------------------

This file demonstrates:
- Extracting feature importances from decision trees
- Extracting feature importances from Random Forests
- Visualizing feature importances
- Understanding how feature importance is calculated

Run:
    python Day-6-Decision-Trees-Random-Forests/feature_importance_analysis. py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn. datasets import load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def basic_feature_importance():
    """
    Extract and display basic feature importances.
    
    Feature importance in trees:
    - Measures how much each feature contributes to reducing impurity
    - Sum of importances = 1. 0
    - Higher value = more important feature
    """
    print("="*60)
    print("Basic Feature Importance")
    print("="*60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    
    importances = tree.feature_importances_
    
    print("\nFeature Importances (Decision Tree):")
    print(f"Sum of importances: {importances.sum():.4f}")
    
    indices = np.argsort(importances)[::-1][:10]
    
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    print("\nHow it's calculated:")
    print("  - Each split reduces impurity (entropy or gini)")
    print("  - Feature importance = weighted sum of impurity reductions")
    print("  - Features used in top splits get higher importance")


def compare_tree_vs_forest_importance():
    """
    Compare feature importances from single tree vs Random Forest.
    Random Forest importances are more stable (averaged over many trees).
    """
    print("\n" + "="*60)
    print("Single Tree vs Random Forest Feature Importance")
    print("="*60)
    
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X, y)
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest. fit(X, y)
    
    tree_importances = tree.feature_importances_
    forest_importances = forest.feature_importances_
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Tree': tree_importances,
        'Forest': forest_importances
    }). sort_values('Forest', ascending=False)
    
    print("\nTop 10 Features:")
    print(df_importance. head(10). to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    
    plt.bar(x - width/2, df_importance. sort_values('Feature')['Tree'], 
            width, label='Single Tree', alpha=0.7)
    plt.bar(x + width/2, df_importance.sort_values('Feature')['Forest'], 
            width, label='Random Forest', alpha=0. 7)
    
    plt. xlabel('Features')
    plt. ylabel('Importance')
    plt.title('Feature Importance: Single Tree vs Random Forest')
    plt.xticks(x, df_importance.sort_values('Feature')['Feature'], rotation=90, ha='right')
    plt. legend()
    plt.tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/feature_importance_comparison.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved plot: {output_path}")
    plt.close()
    
    print("\n📊 Key Insight:")
    print("  - Random Forest importances are more stable (less variance)")
    print("  - Single tree can be biased by random splits")
    print("  - Use Random Forest importances for reliable feature selection")


def visualize_top_features():
    """
    Create a horizontal bar chart of top features.
    """
    print("\n" + "="*60)
    print("Visualizing Top Features")
    print("="*60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    forest = RandomForestClassifier(n_estimators=200, random_state=42)
    forest.fit(X, y)
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances[::-1], align='center', color='steelblue')
    plt. yticks(range(top_n), top_features[::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features (Random Forest)')
    plt.tight_layout()
    
    output_path = "Day-6-Decision-Trees-Random-Forests/top_features.png"
    plt. savefig(output_path, dpi=100)
    print(f"\nSaved plot: {output_path}")
    plt.close()
    
    print(f"\nTop {top_n} features account for {top_importances.sum():.2%} of total importance")


def feature_importance_stability():
    """
    Show that Random Forest feature importances are more stable across runs.
    """
    print("\n" + "="*60)
    print("Feature Importance Stability")
    print("="*60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    n_runs = 10
    tree_importances_runs = []
    forest_importances_runs = []
    
    print(f"\nRunning {n_runs} experiments with different random seeds...")
    
    for seed in range(n_runs):
        tree = DecisionTreeClassifier(random_state=seed)
        tree.fit(X, y)
        tree_importances_runs. append(tree.feature_importances_)
        
        forest = RandomForestClassifier(n_estimators=100, random_state=seed)
        forest.fit(X, y)
        forest_importances_runs.append(forest.feature_importances_)
    
    tree_importances_array = np.array(tree_importances_runs)
    forest_importances_array = np.array(forest_importances_runs)
    
    tree_std = tree_importances_array.std(axis=0). mean()
    forest_std = forest_importances_array.std(axis=0).mean()
    
    print(f"\nAverage standard deviation across features:")
    print(f"  Single Tree:   {tree_std:.6f}")
    print(f"  Random Forest: {forest_std:.6f}")
    
    print("\n📊 Conclusion:")
    print("  Random Forest importances have lower variance across runs")
    print("  → More reliable for feature selection and interpretation")
    
    top_feature_idx = forest_importances_array.mean(axis=0).argmax()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([tree_importances_array[:, top_feature_idx], 
                 forest_importances_array[:, top_feature_idx]],
                labels=['Single Tree', 'Random Forest'])
    plt.ylabel('Importance')
    plt. title(f'Importance Distribution for\n"{feature_names[top_feature_idx]}"')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    top_5_indices = forest_importances_array.mean(axis=0).argsort()[-5:][::-1]
    positions = np.arange(len(top_5_indices))
    
    tree_means = tree_importances_array[:, top_5_indices]. mean(axis=0)
    tree_stds = tree_importances_array[:, top_5_indices].std(axis=0)
    forest_means = forest_importances_array[:, top_5_indices].mean(axis=0)
    forest_stds = forest_importances_array[:, top_5_indices].std(axis=0)
    
    width = 0.35
    plt.bar(positions - width/2, tree_means, width, yerr=tree_stds, 
            label='Single Tree', alpha=0.7, capsize=5)
    plt. bar(positions + width/2, forest_means, width, yerr=forest_stds, 
            label='Random Forest', alpha=0.7, capsize=5)
    
    plt.xlabel('Feature')
    plt.ylabel('Mean Importance')
    plt.title('Top 5 Features: Mean ± Std Dev')
    plt.xticks(positions, [feature_names[i][:15] for i in top_5_indices], 
               rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = "Day-6-Decision-Trees-Random-Forests/importance_stability.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved plot: {output_path}")
    plt.close()


def feature_selection_example():
    """
    Use feature importances for feature selection.
    """
    print("\n" + "="*60)
    print("Feature Selection Using Importances")
    print("="*60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest. fit(X_train, y_train)
    
    baseline_score = forest.score(X_test, y_test)
    print(f"\nBaseline (all {X.shape[1]} features): {baseline_score:.4f}")
    
    importances = forest.feature_importances_
    
    top_k_values = [5, 10, 15, 20]
    
    print("\nPerformance with top-K features:")
    for k in top_k_values:
        top_indices = np.argsort(importances)[::-1][:k]
        
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]
        
        forest_selected = RandomForestClassifier(n_estimators=100, random_state=42)
        forest_selected.fit(X_train_selected, y_train)
        score = forest_selected.score(X_test_selected, y_test)
        
        print(f"  Top {k:2d} features: {score:.4f}")
    
    print("\n📊 Insight:")
    print("  - Often, a small subset of features captures most predictive power")
    print("  - Feature selection reduces complexity and can improve generalization")


def main():
    """
    Run all feature importance examples.
    """
    basic_feature_importance()
    compare_tree_vs_forest_importance()
    visualize_top_features()
    feature_importance_stability()
    feature_selection_example()
    
    print("\n" + "="*60)
    print("Feature Importance Analysis — Complete!")
    print("="*60)
    print("\nNext: Read common_mistakes.txt to learn best practices.")


if __name__ == "__main__":
    main()
