"""
Chapter 4 — Cumulative Distribution Functions
Question: What percentile is your birth weight? How do distributions compare?

What this script builds:
  - CDF class from scratch
  - Percentile and inverse-CDF lookups
  - CDF-based comparison of first vs other babies
  - Generating synthetic data from an empirical CDF

Run: python ch04_cdf.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import first, other, live

COLORS = {
    'first':   '#2196F3',
    'other':   '#4CAF50',
    'all':     '#FF9800',
    'neutral': '#9E9E9E',
}

# ── CDF: the core data structure ───────────────────────────────────────────────

class Cdf:
    """
    Empirical Cumulative Distribution Function.
    Built from observed data; F(x) = fraction of values <= x.
    """

    def __init__(self, values):
        clean = np.array(sorted(v for v in values if not np.isnan(float(v))))
        n = len(clean)
        self.xs = clean
        # Each point's CDF value is its rank / n (1-indexed)
        self.ps = np.arange(1, n + 1) / n

    def prob(self, x: float) -> float:
        """F(x) = P(X <= x) — fraction of data at or below x."""
        # np.searchsorted finds the insertion point; that's the count of values <= x
        idx = np.searchsorted(self.xs, x, side='right')
        return idx / len(self.xs)

    def value(self, p: float) -> float:
        """Inverse CDF: given probability p, return x such that F(x) ≈ p."""
        if p <= 0:
            return self.xs[0]
        if p >= 1:
            return self.xs[-1]
        idx = int(p * len(self.xs))
        return self.xs[idx]

    def median(self) -> float:
        return self.value(0.5)

    def iqr(self) -> float:
        return self.value(0.75) - self.value(0.25)

    def percentile_rank(self, x: float) -> float:
        """What percentile is this value? Returns 0-100."""
        return self.prob(x) * 100

    def sample(self, n: int) -> np.ndarray:
        """Generate n synthetic values matching this empirical CDF."""
        uniform_samples = np.random.uniform(0, 1, size=n)
        return np.array([self.value(u) for u in uniform_samples])


# ── Build CDFs ─────────────────────────────────────────────────────────────────

wgt_cdf_all   = Cdf(live['totalwgt_lb'].dropna().values)
wgt_cdf_first = Cdf(first['totalwgt_lb'].dropna().values)
wgt_cdf_other = Cdf(other['totalwgt_lb'].dropna().values)

prg_cdf_first = Cdf(first['prglngth'].dropna().values)
prg_cdf_other = Cdf(other['prglngth'].dropna().values)

# ── Percentile analysis ────────────────────────────────────────────────────────

print("── Birth Weight: Percentile Summary ───────────────────────────────────")
for label, cdf in [("All babies", wgt_cdf_all),
                    ("First babies", wgt_cdf_first),
                    ("Other babies", wgt_cdf_other)]:
    print(f"\n  {label}:")
    print(f"    25th percentile : {cdf.value(0.25):.2f} lbs")
    print(f"    Median (50th)   : {cdf.median():.2f} lbs")
    print(f"    75th percentile : {cdf.value(0.75):.2f} lbs")
    print(f"    IQR             : {cdf.iqr():.2f} lbs")

# ── Percentile rank examples ───────────────────────────────────────────────────

print("\n── Where Does a Specific Weight Rank? ─────────────────────────────────")
for weight in [6.0, 7.0, 7.5, 8.0, 9.0, 10.0]:
    rank = wgt_cdf_all.percentile_rank(weight)
    print(f"  A baby weighing {weight:.1f} lbs is at the {rank:.1f}th percentile")

# ── Generating synthetic data ──────────────────────────────────────────────────
# Demonstrates that the inverse CDF lets us simulate new data

np.random.seed(42)
synthetic_weights = wgt_cdf_all.sample(1000)
synthetic_cdf = Cdf(synthetic_weights)

print(f"\n── Synthetic Data Check ────────────────────────────────────────────────")
print(f"  Original  median : {wgt_cdf_all.median():.3f} lbs")
print(f"  Synthetic median : {synthetic_cdf.median():.3f} lbs")
print(f"  (should be close — this is what bootstrap sampling is built on)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. CDF of birth weight — all babies
ax = axes[0]
ax.plot(wgt_cdf_all.xs, wgt_cdf_all.ps, color=COLORS['all'], linewidth=2)
# Mark key percentiles
for p, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
    x = wgt_cdf_all.value(p)
    ax.axvline(x, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.axhline(p, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.annotate(f'{label}\n{x:.1f} lbs', xy=(x, p),
                xytext=(x + 0.3, p - 0.08), fontsize=7)
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('CDF')
ax.set_title('CDF: Birth Weight (all babies)')

# 2. CDF comparison: first vs other
ax = axes[1]
ax.plot(prg_cdf_first.xs, prg_cdf_first.ps,
        color=COLORS['first'], linewidth=2, label='First babies')
ax.plot(prg_cdf_other.xs, prg_cdf_other.ps,
        color=COLORS['other'], linewidth=2, label='Other babies')
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('CDF')
ax.set_title('CDF: Pregnancy Length (first vs other)')
ax.legend()
ax.set_xlim(27, 45)

# 3. Original vs synthetic CDF
ax = axes[2]
ax.plot(wgt_cdf_all.xs, wgt_cdf_all.ps,
        color=COLORS['all'], linewidth=2, label='Original data')
ax.plot(synthetic_cdf.xs, synthetic_cdf.ps,
        color=COLORS['neutral'], linewidth=1.5, linestyle='--', label='Synthetic (n=1000)')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('CDF')
ax.set_title('Original vs CDF-generated Synthetic Data')
ax.legend()

plt.tight_layout()
plt.savefig('ch04_cdf.png', dpi=150)
plt.show()
print("\nFigure saved: ch04_cdf.png")
