# Day 4: Logistic Regression & Classification 🎯

## 📚 Overview
Welcome to Day 4 of Machine Learning! Today we explore **Logistic Regression**, a fundamental algorithm for classification tasks. Unlike Linear Regression which predicts continuous values, Logistic Regression predicts discrete categories (like Yes/No, Spam/Not Spam).

---

## 🎯 Topics Covered

### 1. Binary Classification Basics
- Understanding classification vs regression
- Real-world classification problems
- How to represent categorical outcomes numerically

### 2. Sigmoid Function
The **Sigmoid Function** (also called Logistic Function) is the heart of Logistic Regression:

```
σ(z) = 1 / (1 + e^(-z))
```

**Key Properties:**
- ✅ Output range: (0, 1) - perfect for probabilities
- ✅ S-shaped curve
- ✅ When z = 0, σ(z) = 0.5
- ✅ As z → ∞, σ(z) → 1
- ✅ As z → -∞, σ(z) → 0

### 3. Logistic Regression Model
**Hypothesis Function:**
```
h(x) = σ(w^T * x + b) = 1 / (1 + e^(-(w^T * x + b)))
```

Where:
- `w` = weights (parameters)
- `b` = bias term
- `x` = input features
- Output is between 0 and 1 (probability)

**Decision Rule:**
- If h(x) ≥ 0.5 → Predict class 1
- If h(x) < 0.5 → Predict class 0

### 4. Cost Function (Log Loss)
Unlike Mean Squared Error (MSE) used in Linear Regression, Logistic Regression uses **Log Loss** (Binary Cross-Entropy):

```
Cost(h(x), y) = -y * log(h(x)) - (1 - y) * log(1 - h(x))
```

**Why not MSE?**
- MSE creates a non-convex cost function for logistic regression
- Log Loss is convex, ensuring gradient descent finds global minimum

### 5. Decision Boundary
The **decision boundary** is the line (or surface) that separates different classes:
- For 2D: A line
- For 3D: A plane
- For higher dimensions: A hyperplane

### 6. Classification Metrics

#### Confusion Matrix:
```
                Predicted
                0       1
Actual  0      TN      FP
        1      FN      TP
```

- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I Error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II Error)

#### Key Metrics:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- Overall correctness of the model
- ⚠️ Can be misleading with imbalanced datasets

**Precision** = TP / (TP + FP)
- Out of all positive predictions, how many were correct?
- Important when False Positives are costly

**Recall (Sensitivity)** = TP / (TP + FN)
- Out of all actual positives, how many did we find?
- Important when False Negatives are costly

**F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)
- Harmonic mean of Precision and Recall
- Balances both metrics

---

## 📂 Files Structure

```
Day-4-Logistic-Regression/
│
├── README.md                          # This file
├── sigmoid_function.py                # Understanding sigmoid activation
├── logistic_regression_from_scratch.py # Building the algorithm
├── decision_boundary.py               # Visualizing classification boundaries
├── classification_metrics.py          # Evaluation metrics implementation
└── spam_detection.py                  # Real-world application
```

---

## 🚀 Learning Path

### Recommended Order:
1. **sigmoid_function.py** - Understand the core activation function
2. **logistic_regression_from_scratch.py** - Build the complete algorithm
3. **decision_boundary.py** - Visualize how the model separates classes
4. **classification_metrics.py** - Learn to evaluate classification models
5. **spam_detection.py** - Apply to a real-world problem

---

## 🔑 Key Concepts

### When to Use Logistic Regression?
✅ Binary classification problems
✅ Need probability estimates
✅ Linear decision boundary is appropriate
✅ Interpretable model required

### Logistic vs Linear Regression
| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| **Output** | Continuous values | Probabilities (0-1) |
| **Use Case** | Regression | Classification |
| **Cost Function** | MSE | Log Loss |
| **Activation** | None (identity) | Sigmoid |

---

## 💡 Real-World Applications
- 📧 **Email Spam Detection** (Spam vs Not Spam)
- 🏥 **Disease Diagnosis** (Disease vs Healthy)
- 💳 **Credit Default** (Default vs No Default)
- 🎬 **Customer Churn** (Leave vs Stay)
- 🔐 **Fraud Detection** (Fraud vs Legitimate)

---

## 🎓 Tips for Success
1. Always check for class imbalance in your data
2. Feature scaling improves convergence speed
3. Don't rely solely on accuracy - check precision and recall
4. Visualize decision boundaries to understand model behavior
5. Consider the cost of false positives vs false negatives for your problem

---

## 🔗 Connection to Previous Days
- **Day 1**: Feature scaling helps logistic regression converge faster
- **Day 3**: Similar gradient descent optimization, but different cost function

---

## 📊 Next Steps
After mastering Logistic Regression:
- Multi-class classification (Softmax Regression)
- Support Vector Machines (SVMs)
- Decision Trees and Random Forests
- Neural Networks (deep learning)

---

Happy Learning! 🚀
