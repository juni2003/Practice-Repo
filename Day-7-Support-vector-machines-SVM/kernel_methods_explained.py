"""
Kernel Methods in SVM: Visual comparisons and parameter effects.

Includes:
- Visualizing decision boundaries with linear, polynomial, RBF, sigmoid kernels
- Effect of gamma (RBF) and degree (polynomial)
- Kernel selection guidelines
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def kernel_selection_guide():
    guide = """
Kernel Selection Guide
----------------------
Linear:
  - Best for linearly separable data or very high-dimensional sparse data
  - Fast and simpler to interpret

Polynomial:
  - Use when relationships are polynomial-like
  - Degree controls complexity (higher = more complex)

RBF (Gaussian):
  - Strong default for most non-linear problems
  - gamma controls boundary complexity (small = smooth; large = wiggly)

Sigmoid:
  - Similar to neural network behavior
  - Less commonly used; may not be positive semi-definite
"""
    print(guide)


def visualize_kernel_effects():
    datasets = {
        'Circles': make_circles(n_samples=300, noise=0.08, factor=0.3, random_state=42),
        'Moons': make_moons(n_samples=300, noise=0.1, random_state=42),
        'Blobs(3)': make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42),
    }
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    fig, axes = plt.subplots(len(datasets), len(kernels), figsize=(16, 12))

    for r, (name, (X, y)) in enumerate(datasets.items()):
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        for c, kernel in enumerate(kernels):
            if kernel == 'poly':
                clf = SVC(kernel=kernel, degree=3, gamma='scale')
            else:
                clf = SVC(kernel=kernel, gamma='scale')
            clf.fit(Xs, y)

            h = 0.02
            x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
            y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            ax = axes[r, c]
            ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.3)
            ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
            ax.scatter(Xs[:, 0], Xs[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=25)
            ax.set_title(f"{name} | {kernel.upper()} | acc={clf.score(Xs,y):.2f}")
            ax.grid(True, alpha=0.2)

            if c == 0:
                ax.set_ylabel(name)

    plt.suptitle("Kernel Comparison Across Datasets", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_gamma_effect_rbf():
    X, y = make_circles(n_samples=300, noise=0.08, factor=0.3, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    gammas = [0.01, 0.1, 1, 10]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, gamma in enumerate(gammas):
        clf = SVC(kernel='rbf', gamma=gamma)
        clf.fit(Xs, y)

        h = 0.02
        x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
        y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax = axes[i]
        ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        ax.scatter(Xs[:, 0], Xs[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=25)
        ax.set_title(f"RBF gamma={gamma} | acc={clf.score(Xs,y):.2f}")
        ax.grid(True, alpha=0.2)

    plt.suptitle("RBF Gamma Effect", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_poly_degree_effect():
    X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
    Xs = StandardScaler().fit_transform(X)

    degrees = [2, 3, 5, 7]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, d in enumerate(degrees):
        clf = SVC(kernel='poly', degree=d, gamma='scale')
        clf.fit(Xs, y)

        h = 0.02
        x_min, x_max = Xs[:, 0].min() - 1, Xs[:, 0].max() + 1
        y_min, y_max = Xs[:, 1].min() - 1, Xs[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax = axes[i]
        ax.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.3)
        ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
        ax.scatter(Xs[:, 0], Xs[:, 1], c=y, cmap='RdYlBu', edgecolors='k', s=25)
        ax.set_title(f"Poly degree={d} | acc={clf.score(Xs,y):.2f}")
        ax.grid(True, alpha=0.2)

    plt.suptitle("Polynomial Degree Effect", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    kernel_selection_guide()
    visualize_kernel_effects()
    visualize_gamma_effect_rbf()
    visualize_poly_degree_effect()
