"""
Support Vector Machine (SVM) - Linear SVM from scratch using hinge loss + L2 regularization.

Educational implementation:
- Optimizes: J = (1/m) Σ max(0, 1 - y_i * (w·x_i + b)) + (λ/2) ||w||^2
- Uses simple batch gradient descent
- Labels must be in {-1, +1}

For production, prefer sklearn.svm.SVC or LinearSVC.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


class LinearSVMFromScratch:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.is_fitted = False

    def _compute_cost(self, X, y):
        """
        Hinge loss + L2 regularization.
        y in {-1, +1}
        """
        margins = 1 - y * (X @ self.weights + self.bias)
        hinge_losses = np.maximum(0, margins)
        empirical_loss = np.mean(hinge_losses)
        regularization = (self.lambda_param / 2) * np.dot(self.weights, self.weights)
        return empirical_loss + regularization

    def fit(self, X, y):
        """
        Batch gradient descent on hinge loss objective.
        """
        n_samples, n_features = X.shape
        y = np.where(y <= 0, -1, 1)  # ensure {-1, +1}

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for it in range(self.n_iterations):
            # Compute margins and indicator of violations
            margins = 1 - y * (X @ self.weights + self.bias)
            violations = margins > 0  # boolean mask

            # Gradients
            # d/dw: lambda*w - (1/m) Σ_{violations} y_i x_i
            grad_w = self.lambda_param * self.weights - np.mean((y[violations, None] * X[violations]), axis=0) if np.any(violations) else self.lambda_param * self.weights
            # d/db: - (1/m) Σ_{violations} y_i
            grad_b = -np.mean(y[violations]) if np.any(violations) else 0.0

            # Update
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # Track cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            if self.verbose and (it + 1) % max(1, (self.n_iterations // 10)) == 0:
                print(f"Iter {it+1}/{self.n_iterations} - Cost: {cost:.4f}")

        self.is_fitted = True
        return self

    def decision_function(self, X):
        return X @ self.weights + self.bias

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)  # 1 if >= 0 else 0

    def plot_cost(self, save_path=None):
        plt.figure(figsize=(8, 4))
        plt.plot(self.cost_history, lw=2)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Linear SVM Training Cost")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
        plt.show()


def demo_binary_classification():
    # Synthetic 2D dataset
    X, y = make_classification(
        n_samples=400,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = LinearSVMFromScratch(learning_rate=0.05, lambda_param=0.01, n_iterations=1000, verbose=True)
    svm.fit(X_train_s, y_train)

    y_pred = svm.predict(X_test_s)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    svm.plot_cost()

    # Plot decision boundary
    h = 0.02
    x_min, x_max = X_train_s[:, 0].min() - 1, X_train_s[:, 0].max() + 1
    y_min, y_max = X_train_s[:, 1].min() - 1, X_train_s[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = svm.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k', s=40)
    plt.title("Linear SVM (Scratch) Decision Boundary")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_binary_classification()
