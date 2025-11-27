"""
Generate sample CSV datasets for Day 5 examples. 

This script creates realistic toy datasets with:
- Missing values
- Categorical and numeric features
- Class imbalance (for classification)

Run:
    python Day-5-Data-Preprocessing/data/generate_sample_data. py

Output:
    - sample_classification.csv
    - sample_regression.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_classification_dataset(n_samples=1000, random_state=42):
    """
    Generate a classification dataset with realistic features:
    - age (numeric, some missing)
    - income (numeric, some missing)
    - education (categorical: High School, Bachelor, Master, PhD)
    - city (categorical: multiple cities)
    - loan_approved (target: binary)
    """
    rng = np.random.default_rng(seed=random_state)

    age = rng.normal(loc=40, scale=12, size=n_samples). clip(18, 80)
    income = rng. lognormal(mean=10. 5, sigma=0.8, size=n_samples).clip(20000, 500000)

    education = rng.choice(
        ["High School", "Bachelor", "Master", "PhD"],
        size=n_samples,
        p=[0.3, 0.4, 0.2, 0.1],
    )

    city = rng.choice(
        ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Other"],
        size=n_samples,
        p=[0.2, 0.18, 0.15, 0.12, 0.1, 0.25],
    )

    income_scaled = (income - income.min()) / (income.max() - income.min())
    age_scaled = (age - age. min()) / (age.max() - age.min())

    loan_approved_prob = 0.3 + 0.4 * income_scaled + 0.2 * age_scaled
    loan_approved_prob = loan_approved_prob.clip(0. 1, 0.9)
    loan_approved = rng.binomial(n=1, p=loan_approved_prob, size=n_samples)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "education": education,
            "city": city,
            "loan_approved": loan_approved,
        }
    )

    missing_age_idx = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    missing_income_idx = rng.choice(n_samples, size=int(0.08 * n_samples), replace=False)
    missing_city_idx = rng.choice(n_samples, size=int(0.03 * n_samples), replace=False)

    df. loc[missing_age_idx, "age"] = np.nan
    df.loc[missing_income_idx, "income"] = np.nan
    df.loc[missing_city_idx, "city"] = np.nan

    return df


def generate_regression_dataset(n_samples=800, random_state=42):
    """
    Generate a regression dataset predicting house prices:
    - area_sqft (numeric)
    - bedrooms (numeric, discrete)
    - neighborhood (categorical)
    - age_years (numeric, some missing)
    - has_garage (categorical: Yes/No)
    - price (target: continuous)
    """
    rng = np.random.default_rng(seed=random_state)

    area_sqft = rng.normal(loc=1800, scale=600, size=n_samples). clip(500, 5000)
    bedrooms = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0. 1, 0.25, 0.35, 0.2, 0.1])
    neighborhood = rng. choice(
        ["Downtown", "Suburb", "Rural"], size=n_samples, p=[0.3, 0.5, 0.2]
    )
    age_years = rng.uniform(0, 50, size=n_samples)
    has_garage = rng. choice(["Yes", "No"], size=n_samples, p=[0. 7, 0.3])

    base_price = 100000 + 150 * area_sqft + 20000 * bedrooms - 1000 * age_years
    neighborhood_effect = np.where(
        neighborhood == "Downtown", 50000, np.where(neighborhood == "Suburb", 20000, -10000)
    )
    garage_effect = np.where(has_garage == "Yes", 15000, 0)

    price = base_price + neighborhood_effect + garage_effect
    price += rng.normal(0, 30000, size=n_samples)
    price = price.clip(50000, 1000000)

    df = pd. DataFrame(
        {
            "area_sqft": area_sqft,
            "bedrooms": bedrooms,
            "neighborhood": neighborhood,
            "age_years": age_years,
            "has_garage": has_garage,
            "price": price,
        }
    )

    missing_age_idx = rng.choice(n_samples, size=int(0.06 * n_samples), replace=False)
    missing_garage_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)

    df.loc[missing_age_idx, "age_years"] = np.nan
    df.loc[missing_garage_idx, "has_garage"] = np.nan

    return df


def main():
    output_dir = Path("Day-5-Data-Preprocessing/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample datasets...")

    df_class = generate_classification_dataset()
    df_reg = generate_regression_dataset()

    class_path = output_dir / "sample_classification.csv"
    reg_path = output_dir / "sample_regression.csv"

    df_class.to_csv(class_path, index=False)
    df_reg.to_csv(reg_path, index=False)

    print(f"Saved: {class_path} ({len(df_class)} rows)")
    print(f"Saved: {reg_path} ({len(df_reg)} rows)")

    print("\nClassification dataset preview:")
    print(df_class.head())
    print(f"\nMissing values:\n{df_class.isnull().sum()}")

    print("\nRegression dataset preview:")
    print(df_reg.head())
    print(f"\nMissing values:\n{df_reg. isnull().sum()}")


if __name__ == "__main__":
    main()
