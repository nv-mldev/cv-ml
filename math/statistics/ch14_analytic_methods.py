"""
Chapter 14 — Analytic Methods
Question: When can we use formulas instead of simulation?

What this script builds:
  - CLT demonstration: watch the sampling distribution converge to normal
  - Two-sample t-test vs permutation test (compare answers)
  - Analytic confidence interval vs bootstrap CI
  - Analytic correlation test vs simulation

Run: python ch14_analytic_methods.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import first, other, live
from ch09_hypothesis_testing import permutation_test, diff_means

COLORS = {
    'data':     '#2196F3',
    'normal':   '#F44336',
    'small_n':  '#FF9800',
    'large_n':  '#4CAF50',
    'neutral':  '#9E9E9E',
}

np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────
weights = live['totalwgt_lb'].dropna().values
weights = weights[(weights > 0) & (weights < 20)]

first_prg = first['prglngth'].dropna().values
other_prg = other['prglngth'].dropna().values

# ── CLT Demonstration ─────────────────────────────────────────────────────────
# Population: birth weight — slightly left-skewed, NOT normal.
# Sample means of size n should converge to normal as n grows.

sample_sizes = [5, 20, 100, 500]
n_samples = 5000
sampling_distributions = {}

print("── Central Limit Theorem: Birth Weight ─────────────────────────────────")
print(f"  Population: mean={weights.mean():.3f}, std={weights.std():.3f}, "
      f"skew={stats.skew(weights):.3f}")
print(f"\n  {'n':>6}  {'Mean of means':>14}  {'Std of means':>14}  {'σ/√n (theory)':>14}  {'Skew of means':>14}")

for n in sample_sizes:
    sample_means = [np.random.choice(weights, size=n).mean() for _ in range(n_samples)]
    sample_means = np.array(sample_means)
    sampling_distributions[n] = sample_means
    theory_se = weights.std() / np.sqrt(n)
    print(f"  {n:>6}  {sample_means.mean():>14.4f}  {sample_means.std():>14.4f}"
          f"  {theory_se:>14.4f}  {stats.skew(sample_means):>14.4f}")

print(f"\n  As n grows: std(means) → σ/√n, skewness → 0 (converges to normal)")

# ── t-test vs permutation test ─────────────────────────────────────────────────

def two_sample_ttest(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """
    Welch's two-sample t-test (no equal-variance assumption).
    Returns (t_statistic, p_value).
    """
    n1, n2 = len(group1), len(group2)
    m1, m2 = group1.mean(), group2.mean()
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t = (m1 - m2) / se
    # Welch-Satterthwaite degrees of freedom
    num = (s1**2/n1 + s2**2/n2)**2
    den = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df = num / den
    # Two-tailed p-value from t-distribution
    p = 2 * stats.t.sf(abs(t), df=df)
    return t, p


t_stat, p_ttest = two_sample_ttest(first_prg, other_prg)
obs_diff, p_perm, _ = permutation_test(first_prg, other_prg, diff_means, n_permutations=2000)

print(f"\n── t-test vs Permutation Test: Pregnancy Length ────────────────────────")
print(f"  Observed difference : {obs_diff:.4f} weeks")
print(f"  t-test  p-value     : {p_ttest:.6f}")
print(f"  Permutation p-value : {p_perm:.4f}")
print(f"  Same conclusion? {'YES' if (p_ttest < 0.05) == (p_perm < 0.05) else 'NO'}")
print(f"  (With large n, both methods converge — CLT validates the t-test)")

# ── Analytic CI vs bootstrap CI ───────────────────────────────────────────────

n = len(weights)
mean_w = weights.mean()
se_w   = weights.std() / np.sqrt(n)

# Analytic 95% CI using normal approximation (CLT)
z = stats.norm.ppf(0.975)   # 1.96
ci_analytic = (mean_w - z * se_w, mean_w + z * se_w)

# Bootstrap CI
boot_means = np.array([np.random.choice(weights, size=n, replace=True).mean()
                        for _ in range(3000)])
ci_bootstrap = (np.percentile(boot_means, 2.5), np.percentile(boot_means, 97.5))

print(f"\n── Confidence Intervals: Mean Birth Weight ─────────────────────────────")
print(f"  Sample mean         : {mean_w:.4f} lbs")
print(f"  Standard error      : {se_w:.6f} lbs")
print(f"  Analytic 95% CI     : [{ci_analytic[0]:.4f}, {ci_analytic[1]:.4f}]")
print(f"  Bootstrap 95% CI    : [{ci_bootstrap[0]:.4f}, {ci_bootstrap[1]:.4f}]")
print(f"  Width analytic      : {ci_analytic[1]-ci_analytic[0]:.6f} lbs")
print(f"  Width bootstrap     : {ci_bootstrap[1]-ci_bootstrap[0]:.6f} lbs")
print(f"  (With n={n:,}, both methods agree to 4+ decimal places)")

# ── Analytic correlation test ──────────────────────────────────────────────────

age    = live['agepreg'].dropna().values
weight_all = live['totalwgt_lb'].dropna().values

# align
mask = ~(np.isnan(age) | np.isnan(weight_all))
age_c, wgt_c = age[mask], weight_all[mask]
age_c = age_c[(wgt_c > 0) & (wgt_c < 20)]
wgt_c = wgt_c[(wgt_c > 0) & (wgt_c < 20)]

r, p_corr = stats.pearsonr(age_c, wgt_c)
n_corr = len(age_c)
t_corr = r * np.sqrt(n_corr - 2) / np.sqrt(1 - r**2)

print(f"\n── Analytic Correlation Test: Age vs Birth Weight ──────────────────────")
print(f"  Pearson r           : {r:.4f}")
print(f"  t-statistic         : {t_corr:.4f}")
print(f"  p-value             : {p_corr:.6f}")
print(f"  Interpretation      : {'significant' if p_corr < 0.05 else 'not significant'}")
print(f"  (Formula: t = r√(n-2)/√(1-r²) ~ t_{{n-2}} under H₀: ρ=0)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. CLT: sampling distributions for different n
ax = axes[0, 0]
x_range = np.linspace(5, 10, 200)
for n, color, alpha in [(5, COLORS['small_n'], 0.7),
                         (20, COLORS['neutral'], 0.6),
                         (100, COLORS['data'], 0.5),
                         (500, COLORS['large_n'], 0.8)]:
    sm = sampling_distributions[n]
    ax.hist(sm, bins=60, density=True, alpha=alpha, label=f'n={n}')
# Normal overlay for n=500
sm500 = sampling_distributions[500]
ax.plot(x_range, stats.norm.pdf(x_range, sm500.mean(), sm500.std()),
        color='black', linewidth=2, label='Normal limit')
ax.set_xlabel('Sample mean birth weight (lbs)')
ax.set_title('CLT: Sampling Distribution of Mean')
ax.legend(fontsize=7)
ax.set_xlim(5, 10)

# 2. QQ plots: is the sampling distribution normal?
ax = axes[0, 1]
for n, color in [(5, COLORS['small_n']), (100, COLORS['large_n'])]:
    sm = sampling_distributions[n]
    (osm, osr), (slope, intercept, r) = stats.probplot(sm, dist='norm')
    ax.scatter(osm, osr, s=3, alpha=0.3, color=color, label=f'n={n}')
ax.plot([-4, 4], [-4, 4], 'k--', linewidth=1.5, label='Perfect normal')
ax.set_xlabel('Theoretical normal quantiles')
ax.set_ylabel('Sample quantiles')
ax.set_title('QQ Plot: Convergence to Normal')
ax.legend(fontsize=9)

# 3. t-test vs permutation: null distributions side by side
obs_diff_val, p_perm_2, null_perm = permutation_test(first_prg, other_prg,
                                                       diff_means, n_permutations=2000)
ax = axes[1, 0]
ax.hist(null_perm, bins=60, density=True, color=COLORS['neutral'],
        alpha=0.8, label='Permutation null')
# t-distribution null
t_range = np.linspace(-4, 4, 200)
df_approx = min(len(first_prg), len(other_prg)) - 1
ax.plot(t_range * first_prg.std() / np.sqrt(len(first_prg)),
        stats.t.pdf(t_range, df=df_approx) / (first_prg.std() / np.sqrt(len(first_prg))),
        color=COLORS['normal'], linewidth=2, label='t-distribution null')
ax.axvline(obs_diff_val, color='black', linewidth=2, label=f'Observed = {obs_diff_val:.4f}')
ax.set_xlabel('Difference in means (weeks)')
ax.set_title('Permutation Null vs t-Distribution Null')
ax.legend(fontsize=8)

# 4. Bootstrap vs analytic CI
ax = axes[1, 1]
ax.hist(boot_means, bins=60, density=True, color=COLORS['data'], alpha=0.7,
        label='Bootstrap dist of mean')
ax.axvline(mean_w, color='black', linewidth=2)
for lo, hi, label, color in [
    (ci_analytic[0],  ci_analytic[1],  'Analytic CI',  COLORS['normal']),
    (ci_bootstrap[0], ci_bootstrap[1], 'Bootstrap CI', COLORS['large_n']),
]:
    ax.axvspan(lo, hi, alpha=0.15, color=color, label=label)
    ax.axvline(lo, color=color, linestyle='--', linewidth=1.5)
    ax.axvline(hi, color=color, linestyle='--', linewidth=1.5)
ax.set_xlabel('Mean birth weight (lbs)')
ax.set_title('Analytic vs Bootstrap 95% CI\n(they overlap almost exactly)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('ch14_analytic_methods.png', dpi=150)
plt.show()
print("\nFigure saved: ch14_analytic_methods.png")
print("\n── Course Complete ──────────────────────────────────────────────────────")
print("  All 14 chapters of Think Stats — NSFG dataset, built from scratch.")
print("  Next: Statistical Rethinking enrichment (Bayesian perspective).")
