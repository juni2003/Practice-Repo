"""
Multi-class SVM strategies:
- One-vs-Rest (OvR)
- One-vs-One (OvO)

Scikit-learn handles multi-class internally; here we demonstrate usage and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def explain_strategies():
    print("""
One-vs-Rest (OvR):
- Train k classifiers (each class vs all others)
- Choose class with highest decision score

One-vs-One (OvO):
- Train k(k-1)/2 classifiers for each pair of classes
- Predict via majority vote
""")


def iris_multiclass_demo():
    iris = load_iris()
    X, y = iris.data, iris.target
    names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # OvR (default for SVC with decision_function_shape='ovr')
    clf_ovr = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr', random_state=42)
    clf_ovr.fit(X_train_s, y_train)
    y_pred_ovr = clf_ovr.predict(X_test_s)
    acc_ovr = accuracy_score(y_test, y_pred_ovr)

    print("Iris OvR Accuracy:", acc_ovr)
    print("OvR Report:\n", classification_report(y_test, y_pred_ovr, target_names=names))

    cm_ovr = confusion_matrix(y_test, y_pred_ovr)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_ovr, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title("Iris OvR Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # OvO
    clf_ovo = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovo', random_state=42)
    clf_ovo.fit(X_train_s, y_train)
    y_pred_ovo = clf_ovo.predict(X_test_s)
    acc_ovo = accuracy_score(y_test, y_pred_ovo)

    print("Iris OvO Accuracy:", acc_ovo)
    print("OvO Report:\n", classification_report(y_test, y_pred_ovo, target_names=names))

    cm_ovo = confusion_matrix(y_test, y_pred_ovo)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_ovo, annot=True, fmt='d', cmap='Greens', xticklabels=names, yticklabels=names)
    plt.title("Iris OvO Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def wine_multiclass_demo():
    data = load_wine()
    X, y = data.data, data.target
    names = data.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr', random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print("Wine OvR Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=names))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=names, yticklabels=names)
    plt.title("Wine OvR Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explain_strategies()
    iris_multiclass_demo()
    wine_multiclass_demo()
