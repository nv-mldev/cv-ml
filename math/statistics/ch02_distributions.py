"""
Chapter 2 — Distributions
Question: What does the data actually look like, and how big is the first-baby effect?

What this script builds:
  - Histograms from scratch (manual binning + counting)
  - Mean, variance, standard deviation
  - Cohen's d effect size
  - Normalized histograms for fair group comparison

Run: python ch02_distributions.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import first, other, live   # reuse the cleaned data from ch01

COLORS = {
    'first':     '#2196F3',
    'other':     '#4CAF50',
    'highlight': '#F44336',
    'neutral':   '#9E9E9E',
}

# ── Build a histogram from scratch ────────────────────────────────────────────
# We do this manually so you can see what np.histogram does internally.

def make_histogram(values: np.ndarray, bins: np.ndarray) -> dict[float, int]:
    """
    Count how many values fall in each bin.
    Returns a dict: {bin_left_edge: count}
    """
    counts: dict[float, int] = {}
    for b in bins[:-1]:
        counts[b] = 0
    for value in values:
        if np.isnan(value):
            continue
        # find which bin this value belongs to
        for i in range(len(bins) - 1):
            if bins[i] <= value < bins[i + 1]:
                counts[bins[i]] += 1
                break
    return counts


def normalize_histogram(hist: dict[float, int]) -> dict[float, float]:
    """Convert raw counts to proportions so groups of different sizes are comparable."""
    total = sum(hist.values())
    return {k: v / total for k, v in hist.items()}


# ── Pregnancy length distributions ────────────────────────────────────────────
prglngth_bins = np.arange(27, 46, 1)   # 27, 28, ..., 45 weeks

first_hist = make_histogram(first['prglngth'].values, prglngth_bins)
other_hist = make_histogram(other['prglngth'].values, prglngth_bins)

first_norm = normalize_histogram(first_hist)
other_norm = normalize_histogram(other_hist)

print("── Pregnancy Length Distribution ──────────────────────────────────────")
print(f"{'Weeks':>6}  {'First':>8}  {'Other':>8}  {'Diff':>8}")
for week in sorted(first_norm):
    f = first_norm.get(week, 0)
    o = other_norm.get(week, 0)
    print(f"  {week:>4.0f}    {f:>7.4f}    {o:>7.4f}    {f-o:>+7.4f}")

# ── Summary statistics ────────────────────────────────────────────────────────

def mean(values: np.ndarray) -> float:
    """Arithmetic mean, ignoring NaN."""
    v = values[~np.isnan(values)]
    return v.sum() / len(v)


def variance(values: np.ndarray, ddof: int = 0) -> float:
    """
    Population variance (ddof=0) or sample variance (ddof=1).
    ddof = delta degrees of freedom; use ddof=1 when estimating from a sample.
    """
    v = values[~np.isnan(values)]
    m = mean(v)
    return np.sum((v - m) ** 2) / (len(v) - ddof)


def std(values: np.ndarray, ddof: int = 0) -> float:
    return np.sqrt(variance(values, ddof))


first_prg = first['prglngth'].values
other_prg = other['prglngth'].values

print("\n── Summary Statistics: Pregnancy Length ───────────────────────────────")
print(f"                 First babies    Other babies")
print(f"  n            : {len(first_prg):>12,}   {len(other_prg):>12,}")
print(f"  Mean (weeks) : {mean(first_prg):>12.3f}   {mean(other_prg):>12.3f}")
print(f"  Variance     : {variance(first_prg):>12.3f}   {variance(other_prg):>12.3f}")
print(f"  Std dev      : {std(first_prg):>12.3f}   {std(other_prg):>12.3f}")

# ── Cohen's d ─────────────────────────────────────────────────────────────────
# Effect size: difference in means normalized by pooled standard deviation.
# d < 0.2 is considered small, ~0.5 medium, > 0.8 large.

def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cohen's d: how many standard deviations apart are the two group means?
    Uses pooled standard deviation to account for different sample sizes.
    """
    n1 = len(group1[~np.isnan(group1)])
    n2 = len(group2[~np.isnan(group2)])
    s1 = std(group1, ddof=1)
    s2 = std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (mean(group1) - mean(group2)) / pooled_std


d_prglngth = cohens_d(first_prg, other_prg)
print(f"\n── Effect Size ────────────────────────────────────────────────────────")
print(f"  Cohen's d (pregnancy length) : {d_prglngth:.4f}")
print(f"  Interpretation               : {'small' if abs(d_prglngth) < 0.2 else 'medium' if abs(d_prglngth) < 0.5 else 'large'}")
print(f"\n  The anecdote is real but tiny.")
print(f"  Most first babies arrive at the same time as other babies.")

# Birth weight
first_wgt = first['totalwgt_lb'].dropna().values
other_wgt = other['totalwgt_lb'].dropna().values
d_wgt = cohens_d(first_wgt, other_wgt)
print(f"\n  Cohen's d (birth weight)     : {d_wgt:.4f}")
print(f"  Interpretation               : {'small' if abs(d_wgt) < 0.2 else 'medium' if abs(d_wgt) < 0.5 else 'large'}")
print(f"  First babies are slightly lighter — opposite of the late-birth story!")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Normalized histogram: pregnancy length
ax = axes[0]
weeks = sorted(first_norm.keys())
width = 0.4
ax.bar([w - 0.2 for w in weeks], [first_norm[w] for w in weeks],
       width=width, color=COLORS['first'], alpha=0.8, label='First babies')
ax.bar([w + 0.2 for w in weeks], [other_norm[w] for w in weeks],
       width=width, color=COLORS['other'], alpha=0.8, label='Other babies')
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('Proportion')
ax.set_title('Pregnancy Length (normalized)')
ax.legend(fontsize=8)

# 2. Birth weight histogram
ax = axes[1]
wgt_bins = np.arange(0, 16, 0.5)
ax.hist(first_wgt, bins=wgt_bins, density=True, alpha=0.6,
        color=COLORS['first'], label='First')
ax.hist(other_wgt, bins=wgt_bins, density=True, alpha=0.6,
        color=COLORS['other'], label='Other')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('Density')
ax.set_title('Birth Weight Distribution')
ax.legend()

# 3. Effect sizes comparison
ax = axes[2]
effects = [d_prglngth, d_wgt]
labels = ['Pregnancy\nlength', 'Birth\nweight']
bar_colors = [COLORS['first'] if d >= 0 else COLORS['highlight'] for d in effects]
bars = ax.bar(labels, [abs(d) for d in effects], color=bar_colors, alpha=0.8)
ax.axhline(0.2, color='grey', linestyle='--', linewidth=1, label='Small threshold (0.2)')
ax.set_ylabel("Cohen's d (absolute)")
ax.set_title("Effect Sizes")
ax.legend(fontsize=8)
for bar, d in zip(bars, effects):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f'{d:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('ch02_distributions.png', dpi=150)
plt.show()
print("\nFigure saved: ch02_distributions.png")
