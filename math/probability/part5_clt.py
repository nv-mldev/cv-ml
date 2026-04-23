"""
Part 5: The Central Limit Theorem — Why Everything Becomes Gaussian

Proves the CLT experimentally by summing random variables from four very
non-Gaussian distributions and watching them converge to the bell curve.
Then quantifies convergence rate with the Kolmogorov-Smirnov test.

What this script demonstrates:
  - Uniform, Exponential, Bernoulli, and bimodal distributions all converge
    to Gaussian when summed — regardless of their original shape
  - Convergence is visible by n=10 and tight by n=50
  - KS distance vs n shows O(1/√n) convergence rate
  - Symmetric distributions (Uniform) converge faster than skewed (Exponential)

Run: python part5_clt.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}

np.random.seed(42)
print("Setup complete.")

# ── Algorithm ──────────────────────────────────────────────
# CLT in action — sums of non-Gaussian variables become Gaussian
# 1. Define four very non-Gaussian distributions:
#    Uniform, Exponential, Bernoulli, and custom bimodal
# 2. For each distribution and each n in [1, 2, 10, 50]:
#    - Draw 50,000 samples of size n
#    - Compute the sum (not normalized) of n variables
#    - Plot histogram and overlay the Normal fit
# What to look for:
#   Column 1 (n=1):  Original distribution shapes — very different
#   Column 2 (n=2):  Already smoothing out
#   Column 3 (n=10): Clearly bell-shaped, regardless of original
#   Column 4 (n=50): Practically perfect Gaussian
# This is why sensor noise is Gaussian: sum of many independent effects.
# ───────────────────────────────────────────────────────────

np.random.seed(42)
number_of_samples = 50000

distributions = {
    'Uniform [0, 1]': lambda size: np.random.uniform(0, 1, size),
    'Exponential (λ=1)': lambda size: np.random.exponential(1, size),
    'Bernoulli (p=0.3)': lambda size: np.random.binomial(1, 0.3, size),
    'Custom bimodal': lambda size: np.where(np.random.random(size) < 0.5,
                                             np.random.normal(-3, 0.5, size),
                                             np.random.normal(3, 0.5, size)),
}

fig, axes = plt.subplots(len(distributions), 4, figsize=(20, 14))

n_sums = [1, 2, 10, 50]

for row, (dist_name, sampler) in enumerate(distributions.items()):
    for col, n in enumerate(n_sums):
        # Generate n independent samples and sum them
        total = np.zeros(number_of_samples)
        for _ in range(n):
            total += sampler(number_of_samples)

        ax = axes[row, col]
        ax.hist(total, bins=80, density=True, color=COLORS['primary'], alpha=0.6)

        # Overlay the Normal approximation (CLT prediction)
        x_range = np.linspace(total.min(), total.max(), 300)
        mu_hat = total.mean()
        sigma_hat = total.std()
        ax.plot(x_range, norm.pdf(x_range, mu_hat, sigma_hat),
                color=COLORS['highlight'], linewidth=2)

        if row == 0:
            ax.set_title(f'Sum of {n}', fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=10, fontweight='bold')

        ax.set_yticks([])

plt.suptitle('Central Limit Theorem: ANY Distribution → Gaussian When Summed\n'
             '(blue = actual histogram, red = Gaussian fit)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("Column 1 (n=1):  Original distribution shapes — very different from each other")
print("Column 2 (n=2):  Already smoothing out")
print("Column 3 (n=10): Clearly bell-shaped, regardless of the original distribution")
print("Column 4 (n=50): Practically perfect Gaussian — the CLT in action")
print()
print("This is why sensor noise is Gaussian: it's the sum of many independent random effects.")

# ── Algorithm ──────────────────────────────────────────────
# Quantify CLT convergence rate with the KS test
# 1. For each distribution and each n in [1, 2, 3, 5, 10, 20, 50, 100]:
#    - Draw normalized sums of n variables (standardized to zero mean, unit std)
#    - Run Kolmogorov-Smirnov test against N(0,1)
#    - KS statistic = 0 means identical; 1 means maximally different
# 2. Plot KS distance vs n for all four distributions on log-log scale
# What to look for: all curves decrease but at different rates.
#   Symmetric → Uniform converges fastest.
#   Skewed → Exponential converges slower.
#   Bimodal → slowest (two peaks need the most averaging to merge).
# ───────────────────────────────────────────────────────────

n_values = [1, 2, 3, 5, 10, 20, 50, 100]
dist_fns = {
    'Uniform': lambda size: np.random.uniform(0, 1, size),
    'Exponential': lambda size: np.random.exponential(1, size),
    'Bernoulli(0.3)': lambda size: np.random.binomial(1, 0.3, size),
    'Bimodal': lambda size: np.where(np.random.random(size) < 0.5,
                                      np.random.normal(-3, 0.5, size),
                                      np.random.normal(3, 0.5, size)),
}

fig, ax = plt.subplots(figsize=(12, 6))

for dist_name, sampler in dist_fns.items():
    ks_distances = []
    for n in n_values:
        # Sum n independent samples
        total = np.zeros(10000)
        for _ in range(n):
            total += sampler(10000)

        # Standardise: subtract mean, divide by std
        standardised = (total - total.mean()) / total.std()

        # KS test against standard normal
        ks_stat, _ = kstest(standardised, 'norm')
        ks_distances.append(ks_stat)

    ax.plot(n_values, ks_distances, 'o-', linewidth=2, markersize=6, label=dist_name)

ax.axhline(y=0.02, color='gray', linestyle='--', alpha=0.5, label='≈ Gaussian threshold')
ax.set_xlabel('Number of variables summed (n)', fontsize=12)
ax.set_ylabel('KS distance from Gaussian', fontsize=12)
ax.set_title('CLT Convergence Rate: How Quickly Does the Sum Become Gaussian?', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

plt.tight_layout()
plt.show()

print("All distributions converge to Gaussian, but at different rates:")
print("  - Uniform: fastest (already symmetric, just needs smoothing)")
print("  - Bernoulli: moderate (highly discrete, but symmetric for p ≈ 0.5)")
print("  - Exponential: slower (very skewed, needs more averaging)")
print("  - Bimodal: slowest (two peaks need the most averaging to merge)")
print()
print("For sensors: photon counts (Poisson) converge fast because Poisson")
print("is already unimodal and nearly symmetric for λ > 10.")
