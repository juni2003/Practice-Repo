"""
Multinomial Naive Bayes — Text Classification Pipeline

Covers:
- CountVectorizer and TF-IDF vectorization
- MultinomialNB with smoothing
- Train/test split and evaluation
- Notes on class imbalance

Uses a small synthetic dataset to avoid external downloads.
"""

from __future__ import annotations
import numpy as np
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def build_dataset() -> tuple[list[str], np.ndarray]:
    spam = [
        "Win cash now click here",
        "Limited time offer free prize",
        "Exclusive deal act now",
        "Congratulations you won a lottery",
        "Urgent verify your account for reward",
        "Cheap meds available online",
        "Claim your free gift card"
    ]
    ham = [
        "Meeting scheduled tomorrow at 10am",
        "Please review the attached report",
        "Your order has been shipped",
        "Team standup daily at 9am",
        "Thanks for your feedback on the document",
        "Invoice attached for your records",
        "Lunch with client at 1pm"
    ]
    texts = spam + ham
    y = np.array([1] * len(spam) + [0] * len(ham))
    return texts, y


def run_pipeline(use_tfidf: bool = False) -> None:
    texts, y = build_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.33, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english") if use_tfidf else CountVectorizer(stop_words="english")
    pipe = Pipeline([
        ("vec", vectorizer),
        ("nb", MultinomialNB(alpha=1.0))  # Laplace smoothing
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print(f"Vectorizer: {'TF-IDF' if use_tfidf else 'Count'}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix ({'TF-IDF' if use_tfidf else 'Count'} + MultinomialNB)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("CountVectorizer + MultinomialNB:")
    run_pipeline(use_tfidf=False)
    print("\nTF-IDF + MultinomialNB:")
    run_pipeline(use_tfidf=True)
