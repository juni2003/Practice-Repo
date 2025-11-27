"""
Day 6 — Feature Importance Analysis
------------------------------------

This file demonstrates:
- Extracting feature importances from decision trees
- Extracting feature importances from Random Forests
- Visualizing feature importances
- Understanding how feature importance is calculated

Run:
    python Day-6-Decision-Trees-Random-Forests/feature_importance_analysis. py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def basic_feature_importance():
    """
    Extract and display basic feature importances.
    
    Feature importance in trees:
    - Measures how much each feature contributes to reducing impurity
    - Sum of importances = 1. 0
    - Higher value = more
