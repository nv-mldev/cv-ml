"""
Chapter 5 — Modeling Distributions
Question: Can we describe birth weight with just 2 numbers?

What this script builds:
  - Fit Normal, Exponential, Lognormal, Pareto to NSFG data
  - Normal probability plot (QQ plot) from scratch
  - Log-scale CDF checks for Exponential and Pareto
  - Visual goodness-of-fit comparison

Run: python ch05_modeling.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live
from ch04_cdf import Cdf

COLORS = {
    'data':        '#2196F3',
    'normal':      '#4CAF50',
    'exponential': '#FF9800',
    'lognormal':   '#9C27B0',
    'pareto':      '#F44336',
    'neutral':     '#9E9E9E',
}

# ── Data ──────────────────────────────────────────────────────────────────────

weights = live['totalwgt_lb'].dropna().values
weights = weights[(weights > 0) & (weights < 20)]   # remove implausible values

prg_lengths = live['prglngth'].dropna().values
prg_lengths = prg_lengths[prg_lengths >= 27]        # full-term only

# ── Normal distribution fit ───────────────────────────────────────────────────
# Maximum likelihood estimate for normal: mu = mean, sigma = std

mu_hat    = weights.mean()
sigma_hat = weights.std()

print("── Normal Fit: Birth Weight ────────────────────────────────────────────")
print(f"  mu    (mean) : {mu_hat:.3f} lbs")
print(f"  sigma (std)  : {sigma_hat:.3f} lbs")
print(f"\n  68-95-99.7 rule check:")
for n_sigma in [1, 2, 3]:
    lo = mu_hat - n_sigma * sigma_hat
    hi = mu_hat + n_sigma * sigma_hat
    fraction = np.mean((weights >= lo) & (weights <= hi))
    expected = [0.683, 0.954, 0.997][n_sigma - 1]
    print(f"    within {n_sigma}σ: {fraction:.3f}  (expected {expected:.3f})")

# ── Normal probability plot from scratch ─────────────────────────────────────
# Sort data, compare each point to what a perfect normal would predict.
# If the data is normal, this plot is a straight line.

def normal_probability_plot_data(values: np.ndarray):
    """
    Returns (theoretical_quantiles, sorted_data) for a QQ plot.
    theoretical_quantiles: what a perfect standard normal would produce at each rank.
    """
    n = len(values)
    sorted_data = np.sort(values)
    # ranks: (i+0.5)/n avoids 0 and 1 which give ±infinity
    ranks = (np.arange(n) + 0.5) / n
    theoretical = stats.norm.ppf(ranks)   # inverse normal CDF
    return theoretical, sorted_data

theory_q, sorted_w = normal_probability_plot_data(weights)

# ── Exponential fit ───────────────────────────────────────────────────────────
# For exponential, the MLE of lambda is 1/mean.
# We use inter-pregnancy intervals: time (months) between consecutive pregnancies.
# Approximate by using prglngth as a proxy (not ideal but illustrative).

# Better: use pregnancy index to find inter-pregnancy gaps
# For simplicity, use prglngth deviations as a stand-in
exp_data = np.diff(np.sort(prg_lengths))
exp_data = exp_data[exp_data > 0]

lambda_hat = 1.0 / exp_data.mean() if len(exp_data) > 0 else 1.0
print(f"\n── Exponential Fit (pregnancy length intervals) ────────────────────────")
print(f"  lambda (rate)  : {lambda_hat:.4f}")
print(f"  mean (1/lambda): {1/lambda_hat:.3f}")

# ── Lognormal fit ─────────────────────────────────────────────────────────────
# If X is lognormal, then log(X) is normal.
# Fit normal to log(weights).

log_weights = np.log(weights)
mu_log    = log_weights.mean()
sigma_log = log_weights.std()

print(f"\n── Lognormal Fit: Birth Weight ─────────────────────────────────────────")
print(f"  mu_log    : {mu_log:.4f}")
print(f"  sigma_log : {sigma_log:.4f}")
print(f"  median (e^mu_log): {np.exp(mu_log):.3f} lbs")

# ── Goodness of fit: KS test ─────────────────────────────────────────────────
# Kolmogorov-Smirnov test: is the data consistent with a given distribution?
# D = max difference between empirical CDF and theoretical CDF
# Small D, large p-value → can't reject the model

ks_normal, p_normal     = stats.kstest(weights, 'norm',  args=(mu_hat, sigma_hat))
ks_lognorm, p_lognorm   = stats.kstest(weights, 'lognorm', args=(sigma_log, 0, np.exp(mu_log)))

print(f"\n── Goodness of Fit (KS test) ───────────────────────────────────────────")
print(f"  Normal   : D={ks_normal:.4f},  p={p_normal:.4f}")
print(f"  Lognormal: D={ks_lognorm:.4f}, p={p_lognorm:.4f}")
print(f"  Lower D = better fit. Higher p = data is plausibly from this model.")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Normal probability plot (QQ plot)
ax = axes[0, 0]
ax.scatter(theory_q, sorted_w, alpha=0.3, s=5, color=COLORS['data'], label='Data')
# Reference line through 25th and 75th percentiles
q25, q75 = np.percentile(weights, [25, 75])
t25, t75 = stats.norm.ppf([0.25, 0.75])
slope = (q75 - q25) / (t75 - t25)
intercept = q25 - slope * t25
x_line = np.array([theory_q.min(), theory_q.max()])
ax.plot(x_line, slope * x_line + intercept, color=COLORS['normal'],
        linewidth=2, label='Normal reference')
ax.set_xlabel('Theoretical normal quantiles')
ax.set_ylabel('Sample quantiles (lbs)')
ax.set_title('Normal Probability Plot: Birth Weight')
ax.legend(fontsize=8)

# 2. Empirical CDF vs Normal model CDF
ax = axes[0, 1]
wgt_cdf = Cdf(weights)
x_range = np.linspace(weights.min(), weights.max(), 300)
ax.plot(wgt_cdf.xs, wgt_cdf.ps, color=COLORS['data'], linewidth=2, label='Empirical CDF')
ax.plot(x_range, stats.norm.cdf(x_range, mu_hat, sigma_hat),
        color=COLORS['normal'], linewidth=2, linestyle='--', label='Normal model')
ax.plot(x_range, stats.lognorm.cdf(x_range, sigma_log, 0, np.exp(mu_log)),
        color=COLORS['lognormal'], linewidth=2, linestyle=':', label='Lognormal model')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('CDF')
ax.set_title('Empirical vs Modeled CDFs')
ax.legend(fontsize=8)

# 3. Histogram with normal overlay
ax = axes[1, 0]
ax.hist(weights, bins=40, density=True, color=COLORS['data'],
        alpha=0.6, label='Data')
x_range2 = np.linspace(0, 16, 300)
ax.plot(x_range2, stats.norm.pdf(x_range2, mu_hat, sigma_hat),
        color=COLORS['normal'], linewidth=2, label=f'Normal(μ={mu_hat:.2f}, σ={sigma_hat:.2f})')
ax.plot(x_range2, stats.lognorm.pdf(x_range2, sigma_log, 0, np.exp(mu_log)),
        color=COLORS['lognormal'], linewidth=2, linestyle='--', label='Lognormal')
ax.set_xlabel('Birth weight (lbs)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Birth Weight with Model Fits')
ax.legend(fontsize=8)
ax.set_xlim(0, 16)

# 4. Log-scale check for exponential shape
ax = axes[1, 1]
if len(exp_data) > 10:
    exp_cdf = Cdf(exp_data)
    # Complementary CDF on log scale — should be linear if exponential
    ccdf = 1 - exp_cdf.ps
    mask = ccdf > 0
    ax.semilogy(exp_cdf.xs[mask], ccdf[mask],
                color=COLORS['data'], linewidth=2, label='1 - CDF (data)')
    x_exp = np.linspace(exp_data.min(), exp_data.max(), 100)
    ax.semilogy(x_exp, np.exp(-lambda_hat * x_exp),
                color=COLORS['exponential'], linestyle='--', linewidth=2,
                label=f'Exponential(λ={lambda_hat:.3f})')
    ax.set_xlabel('Value')
    ax.set_ylabel('1 - CDF (log scale)')
    ax.set_title('Exponential Shape Check (log-y)')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'Need more data for\nexponential fit',
            ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('ch05_modeling.png', dpi=150)
plt.show()
print("\nFigure saved: ch05_modeling.png")
