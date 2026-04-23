"""
Chapter 6 — Probability Density Functions
Question: What is the true smooth shape of the birth weight distribution?

What this script builds:
  - Gaussian KDE from scratch
  - Moments: mean, variance, skewness, kurtosis
  - Comparison of histogram / PMF / CDF / PDF for the same data
  - Silverman's bandwidth rule

Run: python ch06_pdf.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live
from ch04_cdf import Cdf

COLORS = {
    'data':    '#2196F3',
    'kde':     '#F44336',
    'normal':  '#4CAF50',
    'neutral': '#9E9E9E',
}

weights = live['totalwgt_lb'].dropna().values
weights = weights[(weights > 0) & (weights < 20)]

# ── Gaussian KDE from scratch ──────────────────────────────────────────────────
# We place a Gaussian kernel at each data point and sum.
# The bandwidth h controls how wide each kernel is.

def gaussian_kernel(x: float, xi: float, h: float) -> float:
    """Standard Gaussian kernel centered at xi with bandwidth h."""
    z = (x - xi) / h
    return np.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * h)


def kde(eval_points: np.ndarray, data: np.ndarray, h: float) -> np.ndarray:
    """
    Kernel Density Estimate at each point in eval_points.
    For each eval point x, sum Gaussian kernels centered at each data point.
    """
    densities = np.zeros(len(eval_points))
    n = len(data)
    for i, x in enumerate(eval_points):
        densities[i] = sum(gaussian_kernel(x, xi, h) for xi in data) / n
    return densities


def silverman_bandwidth(data: np.ndarray) -> float:
    """
    Silverman's rule of thumb for Gaussian KDE bandwidth.
    h = 1.06 * sigma * n^(-1/5)
    Balances bias (oversmoothing) vs variance (undersmoothing).
    """
    return 1.06 * data.std() * len(data) ** (-1 / 5)


h_silverman = silverman_bandwidth(weights)
print(f"── KDE Bandwidth ────────────────────────────────────────────────────────")
print(f"  Silverman bandwidth : {h_silverman:.4f} lbs")

x_eval = np.linspace(0, 16, 200)

# Use scipy for speed on full dataset, but show the from-scratch version on a sample
from scipy.stats import gaussian_kde as scipy_kde
kde_fitted = scipy_kde(weights, bw_method=h_silverman / weights.std())
kde_vals = kde_fitted(x_eval)

# From-scratch KDE on a small sample to verify it matches
sample = weights[:200]
h_sample = silverman_bandwidth(sample)
x_sample_eval = np.linspace(3, 13, 100)
kde_scratch = kde(x_sample_eval, sample, h_sample)

print(f"\n── Scratch KDE check (n=200 sample) ────────────────────────────────────")
print(f"  Peak at : {x_sample_eval[np.argmax(kde_scratch)]:.2f} lbs")
print(f"  (should be near {weights.mean():.2f} lbs)")

# ── Moments ───────────────────────────────────────────────────────────────────

def moment(data: np.ndarray, k: int) -> float:
    """k-th raw moment: E[X^k]."""
    return np.mean(data ** k)


def central_moment(data: np.ndarray, k: int) -> float:
    """k-th central moment: E[(X - mu)^k]."""
    mu = data.mean()
    return np.mean((data - mu) ** k)


def skewness(data: np.ndarray) -> float:
    """Third standardized central moment."""
    mu = data.mean()
    sigma = data.std()
    return np.mean(((data - mu) / sigma) ** 3)


def kurtosis(data: np.ndarray) -> float:
    """Excess kurtosis (fourth standardized moment minus 3)."""
    mu = data.mean()
    sigma = data.std()
    return np.mean(((data - mu) / sigma) ** 4) - 3


print(f"\n── Moments: NSFG Variables ──────────────────────────────────────────────")
variables = {
    'Birth weight (lbs)':       live['totalwgt_lb'].dropna().values,
    'Pregnancy length (weeks)': live['prglngth'].dropna().values,
    "Mother's age (years)":     live['agepreg'].dropna().values,
}

print(f"  {'Variable':<28}  {'Mean':>6}  {'Std':>6}  {'Skew':>7}  {'Kurt':>7}")
for name, data in variables.items():
    data = data[(data > 0) & ~np.isnan(data)]
    print(f"  {name:<28}  {data.mean():>6.2f}  {data.std():>6.2f}"
          f"  {skewness(data):>+7.3f}  {kurtosis(data):>+7.3f}")

print(f"\n  Skewness interpretation:")
print(f"    > 0 : right tail (rare very large values)")
print(f"    < 0 : left tail (rare very small values)")
print(f"    Birth weight < 0 → premature babies pull the left tail down")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. All four representations for birth weight
ax = axes[0, 0]
ax.hist(weights, bins=40, density=True, alpha=0.4,
        color=COLORS['data'], label='Histogram')
ax.plot(x_eval, kde_vals, color=COLORS['kde'], linewidth=2, label='KDE')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('Density')
ax.set_title('Histogram vs KDE: Birth Weight')
ax.legend()

# 2. Effect of bandwidth on KDE shape
ax = axes[0, 1]
for h_factor, style, label in [(0.3, ':', 'h = 0.3× Silverman (noisy)'),
                                 (1.0, '-', f'h = Silverman ({h_silverman:.3f})'),
                                 (3.0, '--', 'h = 3× Silverman (oversmoothed)')]:
    h = h_silverman * h_factor
    kd = scipy_kde(weights, bw_method=h / weights.std())
    ax.plot(x_eval, kd(x_eval), linestyle=style, linewidth=2, label=label)
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('Density')
ax.set_title('Effect of Bandwidth on KDE')
ax.legend(fontsize=7)

# 3. CDF and approximate PDF (derivative of CDF)
ax = axes[1, 0]
wgt_cdf = Cdf(weights)
ax2 = ax.twinx()
ax.plot(wgt_cdf.xs, wgt_cdf.ps, color=COLORS['data'], linewidth=2, label='CDF')
ax2.plot(x_eval, kde_vals, color=COLORS['kde'], linewidth=2, linestyle='--', label='KDE (PDF)')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('CDF', color=COLORS['data'])
ax2.set_ylabel('Density (KDE)', color=COLORS['kde'])
ax.set_title('CDF and PDF: Two Views of Same Distribution')

# 4. Skewness across variables
ax = axes[1, 1]
skew_vals = []
skew_names = []
for name, data in variables.items():
    data = data[(data > 0) & ~np.isnan(data)]
    skew_vals.append(skewness(data))
    skew_names.append(name.split('(')[0].strip())
colors_skew = [COLORS['kde'] if s > 0 else COLORS['data'] for s in skew_vals]
bars = ax.barh(skew_names, skew_vals, color=colors_skew, alpha=0.8)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel('Skewness')
ax.set_title('Skewness: NSFG Variables')
for bar, val in zip(bars, skew_vals):
    ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig('ch06_pdf.png', dpi=150)
plt.show()
print("\nFigure saved: ch06_pdf.png")
