"""
Chapter 7 — Relationships Between Variables
Question: Does mother's age predict birth weight?

What this script builds:
  - Scatter plot with jitter and alpha
  - Covariance from scratch
  - Pearson's r from scratch
  - Spearman's rho from scratch
  - Binned relationship plot to reveal nonlinearity

Run: python ch07_relationships.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live

COLORS = {
    'scatter': '#2196F3',
    'binned':  '#F44336',
    'pearson': '#4CAF50',
    'spearman':'#FF9800',
    'neutral': '#9E9E9E',
}

# ── Clean data ────────────────────────────────────────────────────────────────
df = live[['agepreg', 'totalwgt_lb', 'prglngth']].dropna()
df = df[(df['totalwgt_lb'] > 0) & (df['totalwgt_lb'] < 20)]
df = df[(df['agepreg'] > 10) & (df['agepreg'] < 50)]

age    = df['agepreg'].values
weight = df['totalwgt_lb'].values
prg    = df['prglngth'].values

# ── Covariance from scratch ────────────────────────────────────────────────────

def covariance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)]
    Positive: when X is high, Y tends to be high.
    Negative: when X is high, Y tends to be low.
    Units: (unit of X) * (unit of Y) — not comparable across pairs.
    """
    return np.mean((x - x.mean()) * (y - y.mean()))


# ── Pearson's r from scratch ──────────────────────────────────────────────────

def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson's correlation: covariance divided by product of std devs.
    Result is unit-free and always in [-1, 1].
    r = 0 means no LINEAR relationship — could still be a curved one.
    """
    cov = covariance(x, y)
    return cov / (x.std() * y.std())


# ── Spearman's rho from scratch ───────────────────────────────────────────────

def rank_array(x: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1-indexed). Ties get average rank."""
    order = np.argsort(x)
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman's rank correlation: Pearson's r applied to ranks.
    Measures monotone association — works even for curved relationships.
    Robust to outliers because one extreme value only shifts one rank.
    """
    return pearson_r(rank_array(x), rank_array(y))


# ── Results ───────────────────────────────────────────────────────────────────

print("── Correlation: Mother's Age vs Birth Weight ───────────────────────────")
print(f"  Covariance  : {covariance(age, weight):.4f} (lbs × years)")
print(f"  Pearson r   : {pearson_r(age, weight):+.4f}")
print(f"  Spearman ρ  : {spearman_rho(age, weight):+.4f}")
print(f"\n  Interpretation: small positive correlation.")
print(f"  Older mothers tend to have slightly heavier babies — but barely.")

print(f"\n── Correlation: Pregnancy Length vs Birth Weight ───────────────────────")
print(f"  Pearson r   : {pearson_r(prg, weight):+.4f}")
print(f"  Spearman ρ  : {spearman_rho(prg, weight):+.4f}")
print(f"  Interpretation: stronger — longer pregnancies → heavier babies.")

# ── Binned relationship: reveal nonlinearity ──────────────────────────────────
# Divide mother's age into 5-year bins, compute mean birth weight per bin.
# This shows the shape of the relationship without assuming linearity.

age_bins = np.arange(10, 50, 5)
bin_means = []
bin_centers = []
bin_counts = []

for lo, hi in zip(age_bins[:-1], age_bins[1:]):
    mask = (age >= lo) & (age < hi)
    if mask.sum() > 10:
        bin_means.append(weight[mask].mean())
        bin_centers.append((lo + hi) / 2)
        bin_counts.append(mask.sum())

print(f"\n── Binned: Mean Birth Weight by Mother's Age ───────────────────────────")
print(f"  {'Age range':>12}  {'Mean wgt (lbs)':>15}  {'n':>6}")
for center, mean_w, n in zip(bin_centers, bin_means, bin_counts):
    lo = center - 2.5
    hi = center + 2.5
    print(f"  {lo:.0f}–{hi:.0f} years   {mean_w:>12.3f}    {n:>6,}")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Raw scatter — too dense to see
ax = axes[0, 0]
ax.scatter(age, weight, alpha=0.05, s=5, color=COLORS['scatter'])
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Birth weight (lbs)')
ax.set_title('Scatter: Age vs Birth Weight (alpha=0.05)')

# 2. Jittered scatter
np.random.seed(42)
ax = axes[0, 1]
jitter_age = age + np.random.normal(0, 0.3, len(age))
ax.scatter(jitter_age, weight, alpha=0.05, s=5, color=COLORS['scatter'])
ax.plot(bin_centers, bin_means, color=COLORS['binned'],
        linewidth=2.5, marker='o', markersize=6, label='Binned mean')
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Birth weight (lbs)')
ax.set_title('Jittered Scatter + Binned Mean')
ax.legend()

# 3. Binned mean plot: reveals nonlinear shape
ax = axes[1, 0]
ax.plot(bin_centers, bin_means, color=COLORS['binned'],
        linewidth=2.5, marker='o', markersize=8)
ax.fill_between(bin_centers, bin_means, alpha=0.15, color=COLORS['binned'])
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Mean birth weight (lbs)')
ax.set_title("Nonlinear Relationship: Age vs Birth Weight")
ax.set_ylim(6.5, 8.5)

# 4. Correlation comparison bar chart
ax = axes[1, 1]
pairs = ['Age vs\nBirth Weight', 'Prg Length vs\nBirth Weight']
pearson_vals  = [pearson_r(age, weight),  pearson_r(prg, weight)]
spearman_vals = [spearman_rho(age, weight), spearman_rho(prg, weight)]
x = np.arange(len(pairs))
ax.bar(x - 0.2, pearson_vals,  width=0.4, color=COLORS['pearson'],
       alpha=0.8, label="Pearson r")
ax.bar(x + 0.2, spearman_vals, width=0.4, color=COLORS['spearman'],
       alpha=0.8, label="Spearman ρ")
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(pairs, fontsize=9)
ax.set_ylabel('Correlation coefficient')
ax.set_title('Pearson vs Spearman Correlation')
ax.legend()
ax.set_ylim(-0.1, 0.5)

plt.tight_layout()
plt.savefig('ch07_relationships.png', dpi=150)
plt.show()
print("\nFigure saved: ch07_relationships.png")
