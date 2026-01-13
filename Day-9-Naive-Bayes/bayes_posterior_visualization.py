"""
Visualizing Bayes — Likelihoods, Priors, and Posteriors (1D)

Scenario:
- Two classes y ∈ {0, 1}
- Single feature x ~ N(mean_y, var_y) per class (Gaussian NB)
- Vary priors and see how decision boundary/posterior changes
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def posterior_gaussian_1d(x: np.ndarray, mu0: float, var0: float, mu1: float, var1: float, prior1: float):
    prior0 = 1.0 - prior1
    like0 = norm.pdf(x, loc=mu0, scale=np.sqrt(var0))
    like1 = norm.pdf(x, loc=mu1, scale=np.sqrt(var1))
    num1 = like1 * prior1
    num0 = like0 * prior0
    denom = num0 + num1
    p1 = num1 / denom
    p0 = num0 / denom
    return p0, p1


def main():
    # Class 0: mean=-1, var=1 | Class 1: mean=+1, var=1
    mu0, var0 = -1.0, 1.0
    mu1, var1 = +1.0, 1.0

    xs = np.linspace(-5, 5, 400)

    priors = [0.5, 0.3, 0.7]  # prior for class 1
    plt.figure(figsize=(10, 6))

    for i, p1 in enumerate(priors, 1):
        p0, p1_post = posterior_gaussian_1d(xs, mu0, var0, mu1, var1, prior1=p1)

        plt.subplot(2, 3, i)
        plt.plot(xs, p1_post, label=f"P(y=1 | x), prior1={p1}")
        plt.plot(xs, 1 - p1_post, label="P(y=0 | x)")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title("Posterior Probabilities")

        # Likelihoods for reference
        plt.subplot(2, 3, i + 3)
        like0 = norm.pdf(xs, mu0, np.sqrt(var0))
        like1 = norm.pdf(xs, mu1, np.sqrt(var1))
        plt.plot(xs, like0, label="P(x | y=0)")
        plt.plot(xs, like1, label="P(x | y=1)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.title("Class-Conditional Likelihoods")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
