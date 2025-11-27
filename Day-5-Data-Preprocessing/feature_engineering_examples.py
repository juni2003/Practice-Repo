"""
Day 5 — Feature Selection & Feature Extraction (PCA)
----------------------------------------------------

This file demonstrates:
- Filter methods for feature selection:
    - VarianceThreshold (remove low-variance features)
    - Correlation-based selection
    - SelectKBest with univariate statistical tests
- Embedded methods:
    - L1-based feature selection (Lasso / LogisticRegression with L1)
    - Tree-based feature importance
- Feature extraction with PCA:
    - Dimensionality reduction
    - Explained variance analysis
    - Visualization in 2D

Run:
    python Day-5-Data-Preprocessing/feature_engineering_examples.py

Notes:
- Clean code with detailed comments explaining each technique.
- No emojis in code per your request.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    RFE,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def create_classification_dataset(n_samples=800, n_features=20, n_informative=10, random_state=42):
    """
    Create a synthetic classification dataset with some redundant and uninformative features.
    This helps demonstrate how feature selection techniques work.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=5,
        n_repeated=0,
        n_classes=2,
        random_state=random_state,
        shuffle=False,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


def variance_threshold_example(X, threshold=0.1):
    """
    Filter method: Remove features with low variance. 
    Low variance features typically don't provide much information for discrimination. 

    Args:
        X: feature matrix (numpy array or DataFrame)
        threshold: variance threshold (features with variance below this are removed)

    Returns:
        Transformed data and selector object
    """
    print("\n--- Variance Threshold Example ---")
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)

    print(f"Original number of features: {X.shape[1]}")
    print(f"Features after variance threshold ({threshold}): {X_selected. shape[1]}")
    print(f"Removed {X.shape[1] - X_selected.shape[1]} features with low variance")

    return X_selected, selector


def correlation_based_selection(df, target_col="target", threshold=0.95):
    """
    Filter method: Remove highly correlated features.
    When two features are highly correlated, one is often redundant.

    Strategy:
    - Compute pairwise correlation matrix
    - For each pair with correlation > threshold, drop one feature

    Args:
        df: DataFrame with features and target
        target_col: name of target column
        threshold: correlation threshold

    Returns:
        DataFrame with reduced features
    """
    print("\n--- Correlation-Based Selection ---")
    features = [c for c in df.columns if c != target_col]
    X = df[features]

    corr_matrix = X.corr(). abs()

    # Select upper triangle of correlation matrix (avoid duplicates)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1). astype(bool))

    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Original features: {len(features)}")
    print(f"Features to drop (correlation > {threshold}): {len(to_drop)}")
    print(f"Dropped features: {to_drop[:5]}..." if len(to_drop) > 5 else f"Dropped features: {to_drop}")

    df_reduced = df.drop(columns=to_drop)
    return df_reduced


def select_k_best_example(X, y, k=10):
    """
    Filter method: Select top K features based on univariate statistical tests.
    Uses ANOVA F-statistic (f_classif) for classification.

    This method tests each feature independently with the target. 

    Args:
        X: feature matrix
        y: target vector
        k: number of top features to select

    Returns:
        Transformed data and selector
    """
    print("\n--- SelectKBest Example ---")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)

    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)

    print(f"Selected top {k} features based on F-statistic")
    print(f"Selected feature indices: {selected_indices}")
    print(f"Top feature scores: {np.round(scores[selected_indices], 2)}")

    return X_selected, selector


def l1_based_selection_example(X, y):
    """
    Embedded method: Use L1 regularization (Lasso) for feature selection. 
    L1 penalty drives some coefficients to exactly zero, effectively selecting features.

    Args:
        X: feature matrix (scaled)
        y: target vector

    Returns:
        Selected feature indices and model
    """
    print("\n--- L1-Based Feature Selection (Logistic Regression) ---")

    model = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=42)
    model.fit(X, y)

    coefficients = model.coef_[0]
    selected_features = np.where(coefficients != 0)[0]

    print(f"Total features: {X.shape[1]}")
    print(f"Features selected by L1 regularization: {len(selected_features)}")
    print(f"Selected feature indices: {selected_features}")

    return selected_features, model


def tree_based_selection_example(X, y, threshold=0.01):
    """
    Embedded method: Use tree-based model feature importances.
    Random forests compute feature importances based on split quality.

    Args:
        X: feature matrix
        y: target vector
        threshold: importance threshold (keep features above this)

    Returns:
        Selected feature indices and model
    """
    print("\n--- Tree-Based Feature Selection (Random Forest) ---")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model. fit(X, y)

    importances = model.feature_importances_
    selected_features = np.where(importances > threshold)[0]

    print(f"Total features: {X.shape[1]}")
    print(f"Features with importance > {threshold}: {len(selected_features)}")
    print(f"Top 5 feature importances: {np.round(np.sort(importances)[::-1][:5], 4)}")

    return selected_features, model


def pca_example(X, n_components=2, plot=True):
    """
    Feature extraction: Principal Component Analysis (PCA).
    PCA transforms features into orthogonal components ordered by explained variance.

    Use cases:
    - Dimensionality reduction for visualization (2D or 3D)
    - Reducing feature space while retaining most variance
    - Decorrelating features

    Important: Always scale features before PCA. 

    Args:
        X: feature matrix (should be scaled)
        n_components: number of principal components
        plot: whether to plot explained variance

    Returns:
        Transformed data and PCA object
    """
    print("\n--- PCA Feature Extraction ---")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    print(f"Original feature dimension: {X.shape[1]}")
    print(f"Reduced to {n_components} principal components")
    print(f"Explained variance by each component: {np.round(explained_variance, 4)}")
    print(f"Cumulative explained variance: {np.round(cumulative_variance, 4)}")

    if plot and n_components >= 2:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0. 7, color="steelblue")
        plt. xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance by Component")
        plt. xticks(range(1, n_components + 1))

        plt.subplot(1, 2, 2)
        plt.plot(range(1, n_components + 1), cumulative_variance, marker="o", color="darkorange")
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.grid(alpha=0.3)
        plt.xticks(range(1, n_components + 1))

        plt.tight_layout()
        plt.savefig("Day-5-Data-Preprocessing/pca_explained_variance.png", dpi=100)
        print("Saved plot: Day-5-Data-Preprocessing/pca_explained_variance. png")
        plt.close()

    return X_pca, pca


def pca_full_variance_analysis(X):
    """
    Analyze how many components are needed to explain a certain percentage of variance.
    Useful for deciding how many components to keep.

    Args:
        X: scaled feature matrix
    """
    print("\n--- PCA Variance Analysis (All Components) ---")

    pca = PCA()
    pca.fit(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    for threshold in [0.80, 0.90, 0.95, 0.99]:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        print(f"Components needed for {int(threshold*100)}% variance: {n_components}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
    plt.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA: Cumulative Explained Variance")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("Day-5-Data-Preprocessing/pca_cumulative_variance.png", dpi=100)
    print("Saved plot: Day-5-Data-Preprocessing/pca_cumulative_variance.png")
    plt.close()


def run_all_examples():
    """
    Run all feature selection and extraction examples on synthetic dataset.
    """
    print("="*60)
    print("Day 5: Feature Engineering Examples")
    print("="*60)

    df = create_classification_dataset(n_samples=800, n_features=20, n_informative=10)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]. values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    variance_threshold_example(X_train_scaled, threshold=0.1)

    df_reduced = correlation_based_selection(df, target_col="target", threshold=0.85)

    select_k_best_example(X_train_scaled, y_train, k=10)

    l1_based_selection_example(X_train_scaled, y_train)

    tree_based_selection_example(X_train_scaled, y_train, threshold=0.02)

    pca_example(X_train_scaled, n_components=5, plot=True)

    pca_full_variance_analysis(X_train_scaled)

    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    run_all_examples()
