"""
Experiment 1: Missing Values Imputation Strategies
---------------------------------------------------

This experiment compares different imputation strategies:
- Mean imputation
- Median imputation
- Most frequent (mode) imputation
- Constant value imputation
- KNN imputation (uses neighboring samples)

Shows impact on distribution and downstream model performance. 

Run:
    python Day-5-Data-Preprocessing/experiments/01_missing_values_imputation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def load_or_generate_data():
    """
    Load sample classification data if available, otherwise generate synthetic data.
    """
    data_path = Path("Day-5-Data-Preprocessing/data/sample_classification.csv")

    if data_path.exists():
        print(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating synthetic data...")
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=42)
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y

        rng = np.random.default_rng(42)
        for col in df.columns[:-1]:
            if rng.random() > 0.5:
                missing_idx = rng.choice(len(df), size=int(0.1 * len(df)), replace=False)
                df. loc[missing_idx, col] = np.nan

    return df


def compare_imputation_strategies(df, target_col="target"):
    """
    Compare different imputation strategies and evaluate their effect on model performance.
    """
    print("\n" + "="*60)
    print("Comparing Imputation Strategies")
    print("="*60)

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    print(f"\nOriginal data shape: {X.shape}")
    print(f"Missing values per feature:\n{X.isnull(). sum()}")

    strategies = {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "Most Frequent": SimpleImputer(strategy="most_frequent"),
        "Constant (0)": SimpleImputer(strategy="constant", fill_value=0),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
    }

    results = {}

    for name, imputer in strategies.items():
        print(f"\n--- {name} Imputation ---")

        X_imputed = imputer.fit_transform(X)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

        mean_score = scores.mean()
        std_score = scores.std()

        results[name] = {"mean": mean_score, "std": std_score}

        print(f"Cross-validation accuracy: {mean_score:.4f} (+/- {std_score:.4f})")

    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    for name, res in results.items():
        print(f"{name:20s}: {res['mean']:.4f} (+/- {res['std']:.4f})")

    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds = [results[n]["std"] for n in names]

    plt.bar(names, means, yerr=stds, alpha=0.7, color="steelblue", capsize=5)
    plt.ylabel("Cross-Validation Accuracy")
    plt.xlabel("Imputation Strategy")
    plt.title("Comparison of Imputation Strategies")
    plt.xticks(rotation=15, ha="right")
    plt. grid(axis="y", alpha=0.3)
    plt. tight_layout()

    output_path = Path("Day-5-Data-Preprocessing/experiments/imputation_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100)
    print(f"\nSaved plot: {output_path}")
    plt.close()


def visualize_imputation_effect(df, feature_col):
    """
    Visualize how different imputation methods affect the distribution of a single feature.
    """
    if feature_col not in df.columns:
        print(f"Feature {feature_col} not found, skipping visualization.")
        return

    print(f"\nVisualizing imputation effect on feature: {feature_col}")

    original_data = df[feature_col].dropna(). values

    imputers = {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(original_data, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[0].set_title(f"Original Distribution\n(no missing values)")
    axes[0].set_xlabel(feature_col)
    axes[0].set_ylabel("Frequency")

    for idx, (name, imputer) in enumerate(imputers.items(), start=1):
        X_imputed = imputer.fit_transform(df[[feature_col]])
        axes[idx].hist(X_imputed, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        axes[idx]. set_title(f"{name} Imputation")
        axes[idx].set_xlabel(feature_col)

    plt.tight_layout()
    output_path = Path("Day-5-Data-Preprocessing/experiments/imputation_distribution.png")
    plt.savefig(output_path, dpi=100)
    print(f"Saved plot: {output_path}")
    plt.close()


if __name__ == "__main__":
    df = load_or_generate_data()

    compare_imputation_strategies(df, target_col="target" if "target" in df.columns else df.columns[-1])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        feature_to_visualize = numeric_cols[0]
        visualize_imputation_effect(df, feature_to_visualize)
