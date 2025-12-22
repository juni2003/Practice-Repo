# Day 7: Support Vector Machines (SVM)

## Overview
Support Vector Machines (SVM) are powerful supervised learning algorithms for:
- Classification (binary and multi-class)
- Regression (Support Vector Regression, SVR)
- Outlier detection (One-Class SVM)

They work by finding a decision boundary that maximizes the margin between classes. For non-linear problems, SVMs use kernels to implicitly project data into higher-dimensional spaces where it becomes linearly separable.

## Why SVM?
- Effective in high dimensions and with clear margins
- Memory efficient (uses support vectors)
- Versatile (linear and non-linear kernels)

Limitations:
- Can be slower on very large datasets (training scales with support vectors)
- Harder to interpret than tree-based models
- Sensitive to feature scaling and hyperparameters

## Module Structure
```
Day-7-Support-Vector-Machines/
├── README.md
├── svm_theory_and_concepts.md
├── svm_from_scratch.py
├── kernel_methods_explained.py
├── svm_sklearn_tutorial.py
├── multi_class_svm.py
├── svm_real_world_applications.py
└── common_mistakes_and_best_practices.txt
```

## Prerequisites
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- Feature scaling (critical for SVM)

## Quick Start
```python
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

print("Accuracy:", svm.score(X_test, y_test))
```

## Learning Path
1. Theory: svm_theory_and_concepts.md
2. Implementation: svm_from_scratch.py
3. Kernels: kernel_methods_explained.py
4. Practical usage: svm_sklearn_tutorial.py
5. Multi-class strategies: multi_class_svm.py
6. Real-world examples: svm_real_world_applications.py
7. Pitfalls: common_mistakes_and_best_practices.txt

## Key Concepts
- Maximum margin: SVM chooses the boundary with the largest margin
- Support vectors: Points closest to the boundary that define it
- Kernel trick: Compute similarity in high dimensions without explicit mapping

## Hyperparameters
- C (regularization): Larger C reduces regularization (may overfit), smaller C increases regularization (may underfit)
- gamma (kernel width for RBF/poly/sigmoid): Larger gamma → more complex boundary, smaller gamma → smoother boundary
- kernel: 'linear', 'rbf' (default), 'poly', 'sigmoid'

## Tips
- Always scale features (StandardScaler is typical)
- Start with RBF kernel and tune C, gamma
- Use cross-validation + GridSearchCV for tuning
