"""
Day 5 — Data Preprocessing & Feature Engineering
------------------------------------------------

This file contains an end-to-end, commented example showing:
- synthetic dataset creation (numeric + categorical + missing values)
- preprocessing with ColumnTransformer:
    - numeric: impute + scale
    - categorical: impute + one-hot encode
- building a full pipeline with an estimator
- train/test split and cross-validation
- helper to extract transformed feature names (sklearn compatibility notes)

Run:
    python Day-5-Data-Preprocessing/preprocessing_pipeline.py

Notes:
- No emojis in code/comments per your request.
- Explanations of important concepts are inside comments close to the relevant code.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# ---------- Utility: create a synthetic dataset with numeric + categorical + missing ----------
def create_toy_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Create a toy classification dataset with:
    - numeric features from make_classification
    - two categorical features (one low-cardinality, one with some more categories)
    - inject missing values in numeric and categorical columns

    This synthetic dataset helps demonstrate imputation, encoding, and pipeline usage.
    """
    X_num, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=random_state,
    )
    num_cols = [f"num_{i}" for i in range(X_num.shape[1])]
    df_num = pd.DataFrame(X_num, columns=num_cols)

    # Add categorical columns
    rng = np.random.default_rng(seed=random_state)
    cat_1 = rng.choice(["A", "B", "C"], size=n_samples, p=[0.5, 0.3, 0.2])
    cat_2 = rng.choice(["red", "green", "blue", "yellow"], size=n_samples)

    df = df_num.copy()
    df["cat_small"] = cat_1
    df["cat_med"] = cat_2
    df["target"] = y

    # Inject missing values:
    # - numeric: 5% missing in num_1
    # - categorical: 7% missing in cat_med
    n_missing_num = int(0.05 * n_samples)
    n_missing_cat = int(0.07 * n_samples)
    missing_num_idx = rng.choice(n_samples, size=n_missing_num, replace=False)
    missing_cat_idx = rng.choice(n_samples, size=n_missing_cat, replace=False)

    df.loc[missing_num_idx, "num_1"] = np.nan
    df.loc[missing_cat_idx, "cat_med"] = np.nan

    return df

# ---------- Build preprocessing pipeline ----------
def build_preprocessing_pipeline(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
    - numeric: SimpleImputer(strategy='median') -> StandardScaler()
      (median is robust to outliers; scaling needed for many models / PCA)
    - categorical: SimpleImputer(strategy='constant', fill_value='missing')
                   -> OneHotEncoder(handle_unknown='ignore', sparse=False)
    Explanation:
    - We impute categorical missing values with a dedicated token ('missing') so that
      they become a valid category during encoding.
    - handle_unknown='ignore' avoids errors if a category appears in test fold not seen in training.
    """
    # Numeric pipeline
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",  # drop any other columns not specified
    )

    return preprocessor

# Helper: Extract feature names after ColumnTransformer (works with sklearn >= 1.0)
def get_feature_names(column_transformer: ColumnTransformer, input_features: List[str]) -> List[str]:
    """
    Return feature names after a ColumnTransformer.
    Note: Different sklearn versions expose different APIs. This helper tries the
    most common ways. If unavailable, it returns a placeholder list.
    """
    feature_names = []
    # For numeric part, names are the input feature names for numeric_features
    for name, trans, cols in column_transformer.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if hasattr(trans, "named_steps"):
            # Pipeline case
            last_step = trans.named_steps[list(trans.named_steps.keys())[-1]]
            if hasattr(last_step, "get_feature_names_out"):
                out_names = last_step.get_feature_names_out(cols)
            else:
                # For numeric scaler, just use column names
                out_names = cols
        else:
            # If transformer directly supports get_feature_names_out
            if hasattr(trans, "get_feature_names_out"):
                out_names = trans.get_feature_names_out(cols)
            else:
                out_names = cols
        feature_names.extend(list(out_names))
    # If empty, fallback
    if not feature_names:
        return [f"f_{i}" for i in range(len(input_features))]
    return feature_names

# ---------- Example usage: train/test split and cross-validation ----------
def example_pipeline_run():
    df = create_toy_dataset()
    target = "target"
    feature_cols = [c for c in df.columns if c != target]

    # Identify numeric and categorical columns
    numeric_features = [c for c in df.columns if c.startswith("num_")]
    categorical_features = ["cat_small", "cat_med"]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    X = df[feature_cols]
    y = df[target]

    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)

    # Build a full pipeline with an estimator
    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(solver="liblinear")),
        ]
    )

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit pipeline on training data (preprocessing happens inside pipeline, preventing leakage)
    clf.fit(X_train, y_train)

    # Show transformed feature names and shapes
    try:
        transformed = preprocessor.fit(X_train).transform(X_train)
        names = get_feature_names(preprocessor, feature_cols)
        print(f"Transformed shape: {transformed.shape}")
        print(f"Feature names sample (first 20): {names[:20]}")
    except Exception:
        print("Could not extract transformed feature names using helper (sklearn version).")

    # Evaluate with cross-validation (preprocessing inside pipeline — safe)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy scores: {np.round(scores, 4)}")
    print(f"Mean CV accuracy: {scores.mean():.4f}")

    # Quick test-set score
    test_score = clf.score(X_test, y_test)
    print(f"Test set accuracy: {test_score:.4f}")

if __name__ == "__main__":
    example_pipeline_run()
