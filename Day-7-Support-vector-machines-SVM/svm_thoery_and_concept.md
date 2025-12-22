# SVM Theory and Concepts

## 1) Linear SVM (Separability and Margin)
SVM finds a hyperplane that maximizes the margin between two classes.

Decision function:
```
f(x) = w^T x + b
```

Margin:
```
Margin = 2 / ||w||
```

Hard-margin constraints (linearly separable):
```
y_i (w^T x_i + b) >= 1  for all i
```

Primal optimization (hard-margin):
```
minimize   (1/2) ||w||^2
subject to y_i (w^T x_i + b) >= 1
```

Soft-margin (with slack variables ξ_i for non-separable data):
```
minimize   (1/2) ||w||^2 + C * Σ ξ_i
subject to y_i (w^T x_i + b) >= 1 - ξ_i,  ξ_i >= 0
```
- C controls regularization: larger C → less tolerance to violations.

Hinge loss perspective:
```
L_hinge = max(0, 1 - y_i * f(x_i))
Objective J(w,b) = (1/m) Σ L_hinge + (λ/2) ||w||^2
```

## 2) Dual Form and Support Vectors
Dual problem (soft-margin):
```
maximize   Σ α_i - (1/2) ΣΣ α_i α_j y_i y_j K(x_i, x_j)
subject to 0 <= α_i <= C,   Σ α_i y_i = 0
```

The decision function uses support vectors (where α_i > 0):
```
f(x) = Σ α_i y_i K(x_i, x) + b
```

## 3) Kernel Trick
Instead of mapping x → φ(x), compute a kernel:
```
K(x, x') = ⟨φ(x), φ(x')⟩
```
Common kernels:
- Linear: K(x, x') = x^T x'
- Polynomial: K(x, x') = (γ x^T x' + c)^d
- RBF (Gaussian): K(x, x') = exp(-γ ||x - x'||^2)
- Sigmoid: K(x, x') = tanh(γ x^T x' + c)

RBF is a strong default for non-linear problems.

## 4) SVM for Regression (SVR)
SVR uses ε-insensitive loss:
```
minimize (1/2)||w||^2 + C Σ (ξ_i + ξ_i*)
subject to
    y_i - w^T x_i - b <= ε + ξ_i
    w^T x_i + b - y_i <= ε + ξ_i*
    ξ_i, ξ_i* >= 0
```
Predictions aim to be within an ε tube around the regression function, with slack for violations.

## 5) Multi-class SVM
SVM is inherently binary. Extensions:
- One-vs-Rest (OvR): Train k classifiers (class vs others), pick highest score.
- One-vs-One (OvO): Train k(k-1)/2 classifiers (pairs), use majority vote.

Scikit-learn handles this automatically; you can set decision_function_shape='ovr' or 'ovo'.

## 6) When to Use SVM
Use SVM when:
- Clear margin separation exists
- Feature scaling is feasible
- Dataset size is moderate
- Need robust performance on complex boundaries

Avoid SVM when:
- Dataset is extremely large (consider linear models or tree ensembles)
- Need highly interpretable models

## 7) Practical Notes
- Scale features (StandardScaler) before training
- Start with kernel='rbf', gamma='scale', C=1.0
- Use cross-validation to tune C and gamma
- Check support_vectors_ and decision_function for insights
