"""
Experiment 2: Categorical Encoding Comparison
----------------------------------------------

Compares different encoding strategies for categorical variables:
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding (for ordinal features)

Shows when to use each and their impact on model performance.

Run:
    python Day-5-Data-Preprocessing/experiments/02_encoding_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_categorical_dataset(n_samples=800, random_state=42):
    """
    Create a dataset with different types of categorical features:
    - Nominal (no order): color, city
    - Ordinal (has order): education level, satisfaction
    - Binary: has_feature (yes/no)
    """
    rng = np.random. default_rng(seed=random_state)

    color = rng.choice(["red", "blue", "green"], size=n_samples, p=[0.4, 0.35, 0.25])
    city = rng.choice(["NYC", "LA", "Chicago", "Houston"], size=n_samples)

    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n_samples,
        p=[0.3, 0.4, 0. 2, 0.1],
    )

    satisfaction = rng.choice(
        ["Low", "Medium", "High"], size=n_samples, p=[0. 2, 0.5, 0.3]
    )

    has_feature = rng.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    age = rng.normal(35, 10, size=n_samples). clip(18, 70)
    income = rng.lognormal(10. 5, 0.6, size=n_samples).clip(20000, 200000)

    target_prob = 0.1
    target_prob += 0.15 * (color == "red")
    target_prob += 0.10 * (education == "PhD")
    target_prob += 0.20 * (satisfaction == "High")
    target_prob += 0.15 * (has_feature == "Yes")
    target_prob += 0.0001 * income / 10000

    target = rng.binomial(1, target_prob. clip(0.1, 0.9), size=n_samples)

    df = pd.DataFrame(
        {
            "color": color,
            "city": city,
            "education": education,
            "satisfaction": satisfaction,
            "has_feature": has_feature,
            "age": age,
            "income": income,
            "target": target,
        }
    )

    return df


def label_encoding_example(df):
    """
    Demonstrate label encoding (NOT recommended for nominal features with linear models).
    """
    print("\n--- Label Encoding Example ---")

    df_encoded = df.copy()

    for col in ["color", "city", "education", "satisfaction", "has_feature"]:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])

    print("Label encoding applied to all categorical features.")
    print(df_encoded.head())

    X = df_encoded.drop(columns=["target"])
    y = df_encoded["target"]

    model = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"Logistic Regression with Label Encoding: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print("WARNING: Label encoding introduces false ordinal relationships for nominal features.")

    return scores. mean()


def onehot_encoding_example(df):
    """
    Demonstrate one-hot encoding (recommended for nominal features). 
    """
    print("\n--- One-Hot Encoding Example ---")

    nominal_features = ["color", "city", "has_feature"]
    numeric_features = ["age", "income"]

    ordinal_features = ["education", "satisfaction"]
    education_order = [["High School", "Bachelor", "Master", "PhD"]]
    satisfaction_order = [["Low", "Medium", "High"]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("nominal", OneHotEncoder(drop="first", sparse_output=False), nominal_features),
            (
                "ordinal_edu",
                OrdinalEncoder(categories=education_order),
                ["education"],
            ),
            (
                "ordinal_sat",
                OrdinalEncoder(categories=satisfaction_order),
                ["satisfaction"],
            ),
        ]
    )

    X = df. drop(columns=["target"])
    y = df["target"]

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    print(f"Logistic Regression with One-Hot + Ordinal Encoding: {scores. mean():.4f} (+/- {scores.std():.4f})")
    print("One-hot encoding preserves no false ordering for nominal features.")

    return scores. mean()


def tree_model_comparison(df):
    """
    Tree-based models are less sensitive to encoding choice (they can handle label encoding better).
    """
    print("\n--- Tree-Based Model Comparison ---")

    df_label = df.copy()
    for col in ["color", "city", "education", "satisfaction", "has_feature"]:
        le = LabelEncoder()
        df_label[col] = le. fit_transform(df[col])

    X = df_label.drop(columns=["target"])
    y = df_label["target"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"Random Forest with Label Encoding: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print("Tree models can handle label-encoded nominal features without issues.")

    return scores.mean()


def run_all_encoding_experiments():
    """
    Run all encoding experiments and compare results.
    """
    print("="*60)
    print("Categorical Encoding Comparison")
    print("="*60)

    df = create_categorical_dataset()

    print(f"\nDataset shape: {df.shape}")
    print(f"Categorical features: color, city, education, satisfaction, has_feature")
    print(f"Numeric features: age, income")
    print(f"Target: binary classification")

    label_score = label_encoding_example(df)
    onehot_score = onehot_encoding_example(df)
    tree_score = tree_model_comparison(df)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Label Encoding + Logistic Regression:   {label_score:.4f}")
    print(f"One-Hot/Ordinal + Logistic Regression:  {onehot_score:. 4f}")
    print(f"Label Encoding + Random Forest:         {tree_score:. 4f}")
    print("\nKey Takeaway:")
    print("- For linear models, use one-hot encoding for nominal features.")
    print("- Use ordinal encoding only for features with true order.")
    print("- Tree models are more robust to encoding choice.")


if __name__ == "__main__":
    run_all_encoding_experiments()
