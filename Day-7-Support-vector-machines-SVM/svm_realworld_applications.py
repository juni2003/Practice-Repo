"""
Real-world SVM applications:
- Iris and Digits classification
- Spam detection (text features)
- Outlier detection (One-Class SVM)
- End-to-end pipeline examples
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.svm import SVC, OneClassSVM, LinearSVC
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def iris_example():
    iris = load_iris()
    X, y = iris.data, iris.target
    names = iris.target_names

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Iris Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=names, yticklabels=names)
    plt.title("Iris Confusion Matrix (Pipeline)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def digits_example():
    digits = load_digits()
    X, y = digits.data, digits.target

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=5.0, gamma='scale', random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    print("Digits Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title("Digits Confusion Matrix (SVM)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def spam_detection_example():
    spam = [
        "Congratulations! You've won a free prize. Click to claim now.",
        "Get rich quick with this one simple trick.",
        "URGENT: Your account will be closed unless you act now.",
        "Free iPhone for lucky users.",
        "Win cash prizes instantly!"
    ]
    ham = [
        "Meeting scheduled for tomorrow at 10am.",
        "Please review the document and provide feedback.",
        "Your monthly statement is now available.",
        "Thank you for your purchase.",
        "Reminder: Team standup at 9am."
    ]

    texts = spam + ham
    y = np.array([1] * len(spam) + [0] * len(ham))  # 1=spam, 0=ham

    X_train, X_test, y_train, y_test = train_test_split(texts, y, test_size=0.3, random_state=42, stratify=y)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=200)),
        ('clf', LinearSVC(max_iter=3000, random_state=42))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Spam Detection Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])
    plt.title("Spam Detection Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def one_class_outlier_detection_example():
    # Create data: normal cluster + some anomalies
    rng = np.random.RandomState(42)
    normal = 0.3 * rng.randn(300, 2)
    normal = np.r_[normal + 2, normal - 2]
    anomalies = rng.uniform(low=-6, high=6, size=(20, 2))

    clf = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.05)
    clf.fit(normal)

    y_pred_normal = clf.predict(normal)   # +1 inliers, -1 outliers
    y_pred_anom = clf.predict(anomalies)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(normal[:, 0], normal[:, 1], c=(y_pred_normal == 1), cmap='coolwarm', s=20, label='Normal')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c=(y_pred_anom == 1), cmap='coolwarm', s=40, edgecolors='k', label='Anomalies')
    plt.title("One-Class SVM Outlier Detection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    iris_example()
    digits_example()
    spam_detection_example()
    one_class_outlier_detection_example()
