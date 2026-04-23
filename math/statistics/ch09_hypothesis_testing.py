"""
Chapter 9 — Hypothesis Testing
Question: Is the first-baby effect real or just random noise?

What this script builds:
  - Permutation test from scratch (the most honest form of hypothesis testing)
  - Testing difference in means, medians, and Cohen's d
  - p-value vs effect size — why big n makes small effects "significant"
  - Type I error simulation

Run: python ch09_hypothesis_testing.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import first, other, live
from ch02_distributions import cohens_d

COLORS = {
    'null':     '#9E9E9E',
    'observed': '#F44336',
    'reject':   '#FF9800',
    'accept':   '#4CAF50',
}

np.random.seed(42)

# ── Permutation test ───────────────────────────────────────────────────────────

def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic,
    n_permutations: int = 2000,
) -> tuple[float, float, np.ndarray]:
    """
    Permutation test: if labels are arbitrary, shuffle and see how often
    we get a difference as large as the observed one.

    Returns:
        observed_stat   : the actual test statistic
        p_value         : fraction of permutations >= observed
        null_distribution: array of simulated statistics
    """
    observed_stat = statistic(group1, group2)

    pooled = np.concatenate([group1, group2])
    n1 = len(group1)

    null_distribution = np.empty(n_permutations)
    for i in range(n_permutations):
        np.random.shuffle(pooled)
        simulated = statistic(pooled[:n1], pooled[n1:])
        null_distribution[i] = simulated

    # Two-tailed p-value: how often is |simulated| >= |observed|?
    p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_stat))
    return observed_stat, p_value, null_distribution


# ── Test statistics ────────────────────────────────────────────────────────────

def diff_means(a: np.ndarray, b: np.ndarray) -> float:
    return a.mean() - b.mean()


def diff_medians(a: np.ndarray, b: np.ndarray) -> float:
    return np.median(a) - np.median(b)


# ── Run tests: pregnancy length ───────────────────────────────────────────────

first_prg = first['prglngth'].dropna().values
other_prg = other['prglngth'].dropna().values

first_wgt = first['totalwgt_lb'].dropna().values
other_wgt = other['totalwgt_lb'].dropna().values

print("── Permutation Test: Pregnancy Length ──────────────────────────────────")
obs_mean, p_mean, null_mean = permutation_test(first_prg, other_prg, diff_means)
obs_med,  p_med,  null_med  = permutation_test(first_prg, other_prg, diff_medians)

print(f"  Observed diff in means   : {obs_mean:+.4f} weeks")
print(f"  p-value (diff in means)  : {p_mean:.4f}")
print(f"  Observed diff in medians : {obs_med:+.4f} weeks")
print(f"  p-value (diff in medians): {p_med:.4f}")
print(f"  Cohen's d                : {cohens_d(first_prg, other_prg):+.4f}")

print(f"\n  *** KEY INSIGHT ***")
print(f"  p is very small BUT Cohen's d is tiny ({cohens_d(first_prg, other_prg):.4f}).")
print(f"  With n={len(first_prg)+len(other_prg):,}, we can detect effects too small to matter.")
print(f"  Always report effect size alongside p-value.")

print("\n── Permutation Test: Birth Weight ──────────────────────────────────────")
obs_wgt, p_wgt, null_wgt = permutation_test(first_wgt, other_wgt, diff_means)
print(f"  Observed diff in means   : {obs_wgt:+.4f} lbs")
print(f"  p-value                  : {p_wgt:.4f}")
print(f"  Cohen's d                : {cohens_d(first_wgt, other_wgt):+.4f}")
print(f"  First babies are slightly lighter — opposite of the 'born late' story.")

# ── p-value vs sample size ────────────────────────────────────────────────────
# The same tiny effect can be "significant" or not depending purely on n.
# This shows why p-value alone is meaningless without effect size.

print("\n── p-value vs Sample Size ──────────────────────────────────────────────")
print(f"  {'n (subsample)':>15}  {'p-value':>10}  {'Cohen d':>10}")
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    n = int(frac * min(len(first_prg), len(other_prg)))
    sub1 = np.random.choice(first_prg, size=n, replace=False)
    sub2 = np.random.choice(other_prg, size=n, replace=False)
    _, p, _ = permutation_test(sub1, sub2, diff_means, n_permutations=1000)
    d = cohens_d(sub1, sub2)
    sig = "✓ significant" if p < 0.05 else "✗ not significant"
    print(f"  {n:>15,}  {p:>10.4f}  {d:>10.4f}  {sig}")

# ── Type I error simulation ───────────────────────────────────────────────────
# If H0 is TRUE (no real effect), how often do we get p < 0.05?
# Answer: exactly 5% of the time. That is what alpha = 0.05 means.

print("\n── Type I Error Rate Simulation ────────────────────────────────────────")
n_experiments = 500
false_positives = 0
for _ in range(n_experiments):
    # Simulate two groups from the SAME distribution (null is true)
    pool = np.random.normal(0, 1, size=100)
    a = pool[:50]
    b = pool[50:]
    _, p, _ = permutation_test(a, b, diff_means, n_permutations=200)
    if p < 0.05:
        false_positives += 1

type1_rate = false_positives / n_experiments
print(f"  Experiments run       : {n_experiments}")
print(f"  False positives       : {false_positives}")
print(f"  Type I error rate     : {type1_rate:.3f}  (expected ~0.05)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Null distribution: difference in means (pregnancy length)
ax = axes[0]
ax.hist(null_mean, bins=50, density=True, color=COLORS['null'], alpha=0.8,
        label='Null distribution')
ax.axvline(obs_mean, color=COLORS['observed'], linewidth=2,
           label=f'Observed = {obs_mean:.4f}')
ax.axvline(-obs_mean, color=COLORS['observed'], linewidth=2, linestyle='--')
shade = null_mean[np.abs(null_mean) >= np.abs(obs_mean)]
if len(shade) > 0:
    ax.hist(shade, bins=50, density=True, color=COLORS['reject'], alpha=0.7,
            label=f'p = {p_mean:.4f}')
ax.set_xlabel('Difference in means (weeks)')
ax.set_title('Permutation Test: Pregnancy Length')
ax.legend(fontsize=8)

# 2. Null distribution: birth weight
ax = axes[1]
ax.hist(null_wgt, bins=50, density=True, color=COLORS['null'], alpha=0.8,
        label='Null distribution')
ax.axvline(obs_wgt, color=COLORS['observed'], linewidth=2,
           label=f'Observed = {obs_wgt:.4f}')
ax.axvline(-obs_wgt, color=COLORS['observed'], linewidth=2, linestyle='--')
ax.set_xlabel('Difference in means (lbs)')
ax.set_title('Permutation Test: Birth Weight')
ax.legend(fontsize=8)

# 3. p-value vs sample size
ax = axes[2]
ns = []
ps = []
for frac in np.linspace(0.05, 1.0, 20):
    n = max(10, int(frac * min(len(first_prg), len(other_prg))))
    sub1 = np.random.choice(first_prg, size=n, replace=False)
    sub2 = np.random.choice(other_prg, size=n, replace=False)
    _, p, _ = permutation_test(sub1, sub2, diff_means, n_permutations=500)
    ns.append(n)
    ps.append(p)
ax.semilogy(ns, ps, color=COLORS['observed'], linewidth=2, marker='o', markersize=4)
ax.axhline(0.05, color='grey', linestyle='--', linewidth=1, label='p = 0.05')
ax.set_xlabel('Sample size n')
ax.set_ylabel('p-value (log scale)')
ax.set_title('Same Tiny Effect: p Shrinks as n Grows')
ax.legend()

plt.tight_layout()
plt.savefig('ch09_hypothesis_testing.png', dpi=150)
plt.show()
print("\nFigure saved: ch09_hypothesis_testing.png")
