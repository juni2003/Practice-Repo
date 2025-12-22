"""
Scikit-learn SVM Tutorial:
- Basic SVC classification
- Hyperparameter tuning (GridSearchCV)
- Learning curves
- Cross-validation
- Support Vector Regression (SVR)
- LinearSVC feature importance (coefficients)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score


def basic_svc_example():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print("Basic SVC (Iris) Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title("Iris Confusion Matrix (SVC)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def grid_search_tuning():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly']
    }

    gs = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    gs.fit(X_train_s, y_train)

    print("Best Params:", gs.best_params_)
    print("Best CV Score:", gs.best_score_)
    print("Test Accuracy:", gs.score(X_test_s, y_test))

    results = pd.DataFrame(gs.cv_results_)
    top5 = results.nlargest(5, 'mean_test_score')[['param_C', 'param_gamma', 'param_kernel', 'mean_test_score']]
    print("Top 5 combinations:\n", top5.to_string(index=False))


def learning_curves_demo():
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    clf = SVC(kernel='rbf', C=1.0, gamma='auto')

    sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, val_scores = learning_curve(clf, X_train_s, y_train, cv=5, train_sizes=sizes, scoring='accuracy', n_jobs=-1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train', color='blue')
    plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Validation', color='red')
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1),
                     alpha=0.2, color='blue')
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1),
                     alpha=0.2, color='red')
    plt.title("SVM Learning Curves (Digits)")
    plt.xlabel("Train Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cross_validation_demo():
    X, y = load_iris(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    scores = cross_val_score(clf, Xs, y, cv=5, scoring='accuracy')
    print("5-fold CV scores:", scores)
    print("Mean:", scores.mean(), "Std:", scores.std())


def svr_demo():
    X, y = make_regression(n_samples=300, n_features=1, noise=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scalerX = StandardScaler()
    scalerY = StandardScaler()
    X_train_s = scalerX.fit_transform(X_train)
    X_test_s = scalerX.transform(X_test)
    y_train_s = scalerY.fit_transform(y_train.reshape(-1, 1)).ravel()

    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_train_s, y_train_s)

    y_pred_s = svr.predict(X_test_s)
    y_pred = scalerY.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    print("SVR MSE:", mean_squared_error(y_test, y_pred))
    print("SVR R2:", r2_score(y_test, y_pred))

    # Plot fit
    xp = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
    xp_s = scalerX.transform(xp)
    yp_s = svr.predict(xp_s)
    yp = scalerY.inverse_transform(yp_s.reshape(-1, 1)).ravel()

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test, y_test, s=25, alpha=0.6, label="Actual")
    plt.plot(xp, yp, color='red', lw=2, label="SVR fit")
    plt.title("SVR Fit (RBF)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def linear_svc_feature_importance():
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LinearSVC(max_iter=3000, random_state=42)
    clf.fit(X_train_s, y_train)

    print("LinearSVC Test Accuracy:", clf.score(X_test_s, y_test))
    coef = np.abs(clf.coef_[0])
    idx = np.argsort(coef)[::-1][:10]
    print("Top 10 features by |coef|:")
    for i, j in enumerate(idx, 1):
        print(f"{i:2d}. {names[j]:30s} {coef[j]:.4f}")


if __name__ == "__main__":
    basic_svc_example()
    grid_search_tuning()
    learning_curves_demo()
    cross_validation_demo()
    svr_demo()
    linear_svc_feature_importance()
