# 🌳 Day 6 — Decision Trees & Random Forests

Welcome to Day 6!  This module covers one of the most intuitive and powerful machine learning algorithms: **Decision Trees** and their ensemble variant **Random Forests**. 

---

## 📚 Learning Objectives

By the end of this module, you will:

- ✅ Understand decision tree structure (root, nodes, leaves, branches)
- ✅ Calculate **entropy** and **information gain** manually
- ✅ Build a decision tree classifier from scratch
- ✅ Visualize decision trees and decision boundaries
- ✅ Apply tree pruning to prevent overfitting
- ✅ Use Random Forests for ensemble learning
- ✅ Interpret feature importance from trees

---

## 🌲 Key Concepts

### What is a Decision Tree? 

A **decision tree** is a flowchart-like structure where:
- Each **internal node** represents a test on a feature (e.g., "Is age > 30?")
- Each **branch** represents the outcome of that test
- Each **leaf node** represents a class label (classification) or value (regression)

**Example:**

```
                [Age > 30?]
             /               \
          Yes                 No
         /                      \
  [Income > 50k?]            [Student?]
   /        \                 /      \
 Yes        No               Yes      No
 /           \               /          \
Approve    Reject         Reject       Approve
```

### Why Decision Trees? 

✅ **Easy to understand and interpret** — Human-readable rules  
✅ **No feature scaling needed** — Works with raw data  
✅ **Handles non-linear relationships** — Captures complex patterns  
✅ **Feature importance** — Shows which features matter most  
❌ **Prone to overfitting** — Can memorize training data  
❌ **Unstable** — Small data changes can produce very different trees  

---

## 🧮 Core Concepts Explained

### 1.  Entropy

**Entropy** measures the impurity or disorder in a dataset. 

**Formula:**
```
Entropy(S) = -Σ p_i * log₂(p_i)
where `p_i` is the proportion of class `i` in set `S`.
```


**Interpretation:**
- Entropy = 0 → Pure set (all samples belong to one class)
- Entropy = 1 → Maximum impurity (equal distribution of classes)

**Example:**
- Dataset: [Yes, Yes, No, No] → 50% Yes, 50% No
- Entropy = -0.5 * log₂(0. 5) - 0.5 * log₂(0.5) = 1. 0 (maximum impurity)

### 2. Information Gain

**Information Gain** measures how much entropy is reduced after a split.

**Formula:**
```
IG(S, A) = Entropy(S) - Σ (|S_v| / |S|) * Entropy(S_v)

where:
- `S` = parent dataset
- `A` = attribute (feature) to split on
- `S_v` = subset of S where attribute A has value v

```

### 3.  Gini Impurity (Alternative to Entropy)

**Gini Impurity** is another measure of impurity used by sklearn's default decision trees.

**Formula:**
```
Gini(S) = 1 - Σ p_i²

```

**Comparison:**
- Entropy and Gini give similar results
- Gini is computationally faster (no logarithm)
- Entropy penalizes impurity slightly more

### 4. Tree Pruning

**Pruning** reduces tree complexity to prevent overfitting. 

**Pre-pruning (Early Stopping):**
- Set `max_depth` (limit tree depth)
- Set `min_samples_split` (minimum samples to split a node)
- Set `min_samples_leaf` (minimum samples in a leaf)

**Post-pruning (Cost-Complexity Pruning):**
- Grow a full tree, then remove branches that provide little value
- sklearn uses `ccp_alpha` parameter for cost-complexity pruning

### 5. Random Forests

**Random Forest** is an ensemble of many decision trees.

**How it works:**
1. Create multiple decision trees (e.g., 100 trees)
2. Each tree is trained on a **random subset** of data (bootstrap sampling)
3. Each split considers only a **random subset** of features
4.  Final prediction = **majority vote** (classification) or **average** (regression)

**Why Random Forests are better:**
- ✅ Reduces overfitting (averaging reduces variance)
- ✅ More robust and stable
- ✅ Better generalization
- ✅ Can estimate feature importance across all trees

---

## 📁 Files in This Module
```bash
|             File                |                                  Description                               |
|---------------------------------|----------------------------------------------------------------------------|
| `README.md`                     |                  This file — concepts and learning guide                   |
| `decision_tree_basics.py`       |           Introduction to sklearn DecisionTreeClassifier with examples     |
| `entropy_information_gain.py`   |            Manual calculation of entropy and information gain              |
| `decision_tree_from_scratch.py` |              Complete decision tree implementation from scratch            |
| `tree_visualization.py`         |                      Visualize trees and decision boundaries               |
| `tree_pruning_example.py`       |             Demonstrate pre-pruning and post-pruning techniques            |
| `random_forest_ensemble.py`     |              Compare single tree vs Random Forest performance              |
| `feature_importance_analysis.py`|                  Extract and interpret feature importances                 |
| `common_mistakes.txt`           |                     Common pitfalls and best practices                     |

---
```
## 🚀 How to Run Examples

```bash
# Navigate to Day 6 folder
cd Day-6-Decision-Trees-Random-Forests

# Run any example
python decision_tree_basics.py
python entropy_information_gain.py
python decision_tree_from_scratch.py
python tree_visualization.py
python tree_pruning_example.py
python random_forest_ensemble.py
python feature_importance_analysis.py

```

## 🎯 Learning Path
Recommended order:

1. Start with README. md (you are here!) — Understand core concepts
2. decision_tree_basics.py — See sklearn trees in action
3. entropy_information_gain.py — Master the math behind splits
4. decision_tree_from_scratch.py — Understand the algorithm deeply
5. tree_visualization. py — Visualize what trees actually learn
6. tree_pruning_example.py — Learn to control overfitting
7. random_forest_ensemble.py — See the power of ensembles
8. feature_importance_analysis.py — Interpret your models
9. common_mistakes. txt — Avoid common pitfalls


## 🧠 Key Takeaways
-Decision trees split data recursively to maximize information gain (or minimize Gini impurity)
Entropy measures disorder; information gain measures entropy reduction
Trees are prone to overfitting → use pruning or Random Forests
Random Forests improve performance by averaging many trees
Feature importance helps interpret which features drive predictions
No feature scaling needed for tree-based models

## 🔗 Connections to Other Days
Day 1 (Feature Scaling): Trees don't need scaling, but other models do
Day 3 (Linear Regression): Trees capture non-linear patterns better
Day 4 (Logistic Regression): Trees create non-linear decision boundaries
Day 5 (Preprocessing): Trees handle missing values and categorical features naturally (but encoding still helps)



Happy Learning! 🌳🎓


