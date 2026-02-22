# Day 15 — Feature Engineering & Selection

This day focuses on **how to create better features** (feature engineering) and **how to keep only the useful ones** (feature selection), while avoiding the most common trap: **data leakage**.

You’ll get runnable scripts that demonstrate:

- **ColumnTransformer pipelines** for mixed numeric + categorical data
- **Univariate selection**, **RFE**, and **model-based selection**
- **Interaction features** and **target encoding** with leakage guards
- **SHAP interpretation** (global vs local) *(optional dependency)*

---

## Folder Structure

```
Day-15-Feature-Engineering-Selection/
├── README.md
├── columntransformer_pipeline_basics.py
├── selection_univariate_rfe_model_based.py
├── interaction_features_and_polynomials.py
├── target_encoding_with_leakage_guards.py
├── shap_interpretation_global_local.py
└── leakage_cheatsheet.md
```

---

## Installation

Minimum required for most files:
```bash
pip install numpy pandas scikit-learn matplotlib
```

Optional (only for SHAP file):
```bash
pip install shap
```

---

## How to Run

From repo root:

```bash
python Day-15-Feature-Engineering-Selection/columntransformer_pipeline_basics.py
python Day-15-Feature-Engineering-Selection/selection_univariate_rfe_model_based.py
python Day-15-Feature-Engineering-Selection/interaction_features_and_polynomials.py
python Day-15-Feature-Engineering-Selection/target_encoding_with_leakage_guards.py
python Day-15-Feature-Engineering-Selection/shap_interpretation_global_local.py
```

---

## Core Concepts (the “logic”)

### 1) Why Pipelines + ColumnTransformer?
Because you must ensure that:
- **Imputation/Scaling/Encoding are learned only from training data**
- In CV, each fold learns transformations only from its fold’s training split

This prevents leakage and makes your workflow reproducible.

### 2) Feature Selection: three families
1. **Filter methods** (univariate): fast, independent of model  
   Example: `SelectKBest(mutual_info_classif)`
2. **Wrapper methods** (RFE): repeatedly fit model, remove weakest features  
   More accurate but slower.
3. **Embedded methods** (model-based): model selects during training  
   Example: L1 Logistic Regression or tree feature importances.

### 3) Interaction Features
Some models (linear/logistic regression) can’t learn interactions automatically.
Adding `PolynomialFeatures(interaction_only=True)` can unlock performance.

Tree-based models often learn interactions already, but explicit interactions can still help for simpler models.

### 4) Target Encoding (high-cardinality categorical)
Replacing a category with the mean target value can be powerful.
But it is **very easy to leak** if you compute those means using the full dataset.

Safe approach:
- Inside CV folds: compute encoding using **only training fold**, apply to validation fold.
- For final training: fit encoding on full train set, apply to test.

### 5) SHAP (interpretation)
- **Global**: which features matter overall (summary plot)
- **Local**: why did the model predict this for ONE row (waterfall/force plot)

---

## What “Good” Looks Like
- You can rebuild the full preprocessing + selection + model step as **one Pipeline**.
- You validate with cross-validation.
- You never compute target encoding on the full dataset before splitting.
- You can explain which features got selected and why.

---

If you want, tell me your preferred dataset (CSV path) and whether it’s classification or regression; I can adapt Day 15 to your real data with the same leakage-safe patterns.
