"""
Chapter 8 — Estimation
Question: How well does our sample estimate the true population?

What this script builds:
  - Biased vs unbiased variance estimator (simulation)
  - Sampling distribution of the mean
  - Standard error: analytic vs simulated
  - Bootstrap SE for mean and median
  - Effect of survey weights

Run: python ch08_estimation.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live

COLORS = {
    'biased':    '#F44336',
    'unbiased':  '#4CAF50',
    'sample':    '#2196F3',
    'bootstrap': '#FF9800',
    'neutral':   '#9E9E9E',
}

np.random.seed(42)

weights = live['totalwgt_lb'].dropna().values
weights = weights[(weights > 0) & (weights < 20)]

true_mean = weights.mean()
true_std  = weights.std()

# ── Biased vs unbiased variance ───────────────────────────────────────────────
# Simulate: draw many small samples, compute 1/n and 1/(n-1) variance.
# The 1/n version consistently underestimates the true variance.

sample_size = 10
n_simulations = 10_000

biased_vars   = []
unbiased_vars = []

for _ in range(n_simulations):
    sample = np.random.choice(weights, size=sample_size, replace=False)
    mean_s = sample.mean()
    biased_vars.append(np.sum((sample - mean_s) ** 2) / sample_size)
    unbiased_vars.append(np.sum((sample - mean_s) ** 2) / (sample_size - 1))

true_var = weights.var()

print("── Biased vs Unbiased Variance Estimator ───────────────────────────────")
print(f"  True variance (population)   : {true_var:.4f}")
print(f"  Mean of biased estimates     : {np.mean(biased_vars):.4f}  (1/n  → underestimates)")
print(f"  Mean of unbiased estimates   : {np.mean(unbiased_vars):.4f}  (1/n-1 → correct)")
print(f"\n  Bias of 1/n estimator        : {np.mean(biased_vars) - true_var:+.4f}")
print(f"  Bias of 1/(n-1) estimator    : {np.mean(unbiased_vars) - true_var:+.4f}")

# ── Sampling distribution of the mean ────────────────────────────────────────
# Draw many samples of size n=50, compute mean each time.
# The distribution of these means is the sampling distribution.

sample_sizes = [10, 50, 200, 1000]
sampling_stds = {}

print(f"\n── Sampling Distribution of the Mean ───────────────────────────────────")
print(f"  {'n':>6}  {'SE (analytic)':>15}  {'SE (simulated)':>15}")

for n in sample_sizes:
    sample_means = [np.random.choice(weights, size=n).mean() for _ in range(2000)]
    se_analytic  = true_std / np.sqrt(n)
    se_simulated = np.std(sample_means)
    sampling_stds[n] = (sample_means, se_analytic, se_simulated)
    print(f"  {n:>6}  {se_analytic:>15.4f}  {se_simulated:>15.4f}")

print(f"\n  SE halves when n quadruples (SE = sigma / sqrt(n))")

# ── Bootstrap SE ──────────────────────────────────────────────────────────────
# The bootstrap simulates new samples by resampling WITH REPLACEMENT from the data.
# Works for any statistic — not just the mean.

def bootstrap_se(data: np.ndarray, statistic, n_boot: int = 2000) -> float:
    """
    Bootstrap standard error for any statistic.
    Resample with replacement n_boot times, compute statistic each time.
    SE = std of bootstrap distribution.
    """
    estimates = [statistic(np.random.choice(data, size=len(data), replace=True))
                 for _ in range(n_boot)]
    return np.std(estimates), estimates


se_mean,   boot_means   = bootstrap_se(weights, np.mean)
se_median, boot_medians = bootstrap_se(weights, np.median)

print(f"\n── Bootstrap Standard Errors ───────────────────────────────────────────")
print(f"  Mean   : {weights.mean():.4f} lbs  ±  {se_mean:.4f} (SE)")
print(f"  Median : {np.median(weights):.4f} lbs  ±  {se_median:.4f} (SE)")
print(f"  Analytic SE for mean: {true_std / np.sqrt(len(weights)):.4f}")
print(f"  (bootstrap and analytic agree closely for the mean)")

# ── Survey weights ────────────────────────────────────────────────────────────
# NSFG oversampled minority groups. finalwgt corrects for this.
# Ignoring weights biases national estimates.

if 'finalwgt' in live.columns:
    df = live[['totalwgt_lb', 'finalwgt']].dropna()
    df = df[(df['totalwgt_lb'] > 0) & (df['totalwgt_lb'] < 20)]

    unweighted_mean = df['totalwgt_lb'].mean()
    weighted_mean   = np.average(df['totalwgt_lb'], weights=df['finalwgt'])

    print(f"\n── Effect of Survey Weights ────────────────────────────────────────────")
    print(f"  Unweighted mean : {unweighted_mean:.4f} lbs")
    print(f"  Weighted mean   : {weighted_mean:.4f} lbs")
    print(f"  Difference      : {weighted_mean - unweighted_mean:+.4f} lbs")
    print(f"  (weights correct for oversampling — difference is typically small)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Biased vs unbiased variance
ax = axes[0, 0]
ax.hist(biased_vars,   bins=50, density=True, alpha=0.6,
        color=COLORS['biased'],   label='Biased (1/n)')
ax.hist(unbiased_vars, bins=50, density=True, alpha=0.6,
        color=COLORS['unbiased'], label='Unbiased (1/n-1)')
ax.axvline(true_var, color='black', linewidth=2, label=f'True variance = {true_var:.3f}')
ax.set_xlabel('Estimated variance')
ax.set_title(f'Biased vs Unbiased Variance (n={sample_size}, {n_simulations:,} sims)')
ax.legend(fontsize=8)

# 2. Sampling distributions for different n
ax = axes[0, 1]
for n in sample_sizes:
    sample_means, se_a, se_s = sampling_stds[n]
    ax.hist(sample_means, bins=40, density=True, alpha=0.5, label=f'n={n}, SE={se_a:.3f}')
ax.axvline(true_mean, color='black', linewidth=2, label=f'True mean = {true_mean:.3f}')
ax.set_xlabel('Sample mean (lbs)')
ax.set_title('Sampling Distribution of Mean')
ax.legend(fontsize=7)

# 3. SE vs sample size
ax = axes[1, 0]
n_range = np.logspace(1, 4, 50)
se_curve = true_std / np.sqrt(n_range)
ax.loglog(n_range, se_curve, color=COLORS['sample'], linewidth=2)
for n in sample_sizes:
    _, se_a, _ = sampling_stds[n]
    ax.scatter([n], [se_a], s=80, zorder=5)
ax.set_xlabel('Sample size n (log scale)')
ax.set_ylabel('Standard error (log scale)')
ax.set_title('SE = σ/√n  (log-log scale)')

# 4. Bootstrap distribution of mean vs median
ax = axes[1, 1]
ax.hist(boot_means,   bins=40, density=True, alpha=0.6,
        color=COLORS['sample'],    label=f'Bootstrap mean (SE={se_mean:.4f})')
ax.hist(boot_medians, bins=40, density=True, alpha=0.6,
        color=COLORS['bootstrap'], label=f'Bootstrap median (SE={se_median:.4f})')
ax.set_xlabel('Estimate (lbs)')
ax.set_title('Bootstrap Distributions: Mean vs Median')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ch08_estimation.png', dpi=150)
plt.show()
print("\nFigure saved: ch08_estimation.png")
