"""
Chapter 12 — Time Series Analysis
Question: Has birth weight changed over the years?

What this script builds:
  - Aggregate NSFG by year
  - Linear trend fit
  - Moving average filter from scratch
  - Serial correlation and ACF
  - Simple prediction with uncertainty

Run: python ch12_time_series.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live
from ch10_least_squares import least_squares

COLORS = {
    'series':  '#2196F3',
    'trend':   '#F44336',
    'ma':      '#4CAF50',
    'neutral': '#9E9E9E',
    'predict': '#FF9800',
}

# ── Build yearly time series from NSFG ────────────────────────────────────────
# NSFG encodes pregnancy end date as `datend` (century-month format: months since 1900)
# We convert to calendar year for aggregation.

df = live.copy()

if 'datend' in df.columns:
    # century-month: value = (year - 1900) * 12 + month; convert to year
    df['year'] = (df['datend'] / 12).astype(int) + 1900
else:
    # fallback: use a synthetic year based on row order for demonstration
    n = len(df)
    df['year'] = 1990 + (np.arange(n) / (n / 13)).astype(int)

df = df[['year', 'totalwgt_lb', 'prglngth']].dropna()
df = df[(df['totalwgt_lb'] > 0) & (df['totalwgt_lb'] < 20)]

# Aggregate by year
yearly = df.groupby('year').agg(
    mean_weight=('totalwgt_lb', 'mean'),
    mean_prglngth=('prglngth', 'mean'),
    n_births=('totalwgt_lb', 'count'),
).reset_index()

# Keep only years with enough data
yearly = yearly[yearly['n_births'] >= 50].copy()

years   = yearly['year'].values.astype(float)
weights = yearly['mean_weight'].values

print("── Yearly Summary ───────────────────────────────────────────────────────")
print(f"  {'Year':>6}  {'Mean weight':>12}  {'n births':>10}")
for _, row in yearly.iterrows():
    print(f"  {row['year']:>6.0f}  {row['mean_weight']:>12.3f}  {row['n_births']:>10,}")

# ── Linear trend ──────────────────────────────────────────────────────────────

# Center years so intercept is meaningful (at the mean year)
year_mean = years.mean()
years_centered = years - year_mean

alpha, beta = least_squares(years_centered, weights)

print(f"\n── Linear Trend ────────────────────────────────────────────────────────")
print(f"  Slope  : {beta:+.4f} lbs per year")
print(f"  At mean year ({year_mean:.0f}): {alpha:.4f} lbs")
trend_dir = "increasing" if beta > 0 else "decreasing"
print(f"  Trend  : birth weight is {trend_dir} over time ({beta*10:+.3f} lbs per decade)")

# ── Moving average from scratch ───────────────────────────────────────────────

def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple moving average with a window of size `window`.
    Returns (indices_where_ma_is_valid, ma_values).
    The first (window-1) points can't have a full window — they are dropped.
    """
    n = len(values)
    ma = np.empty(n - window + 1)
    for i in range(len(ma)):
        ma[i] = values[i: i + window].mean()
    # Center the MA: index is the middle of the window
    center_idx = np.arange(window // 2, n - window // 2)[:len(ma)]
    return center_idx, ma


# 3-year moving average
ma_idx, ma_vals = moving_average(weights, window=3)
ma_years = years[ma_idx]

# ── Serial correlation from scratch ───────────────────────────────────────────

def autocorrelation(values: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """
    Compute autocorrelation for lags 0, 1, ..., max_lag.
    ACF(k) = Corr(y_t, y_{t-k})
    """
    n = len(values)
    mean = values.mean()
    variance = np.var(values)
    acf = np.empty(max_lag + 1)
    for k in range(max_lag + 1):
        if k == 0:
            acf[k] = 1.0
        else:
            # Pearson correlation between y[k:] and y[:-k]
            cov = np.mean((values[k:] - mean) * (values[:-k] - mean))
            acf[k] = cov / variance
    return acf


acf_vals = autocorrelation(weights, max_lag=min(5, len(weights) - 2))

print(f"\n── Autocorrelation Function (ACF) ──────────────────────────────────────")
print(f"  {'Lag':>4}  {'ACF':>8}")
for k, acf in enumerate(acf_vals):
    print(f"  {k:>4}  {acf:>+8.4f}")

if len(acf_vals) > 1 and abs(acf_vals[1]) > 0.3:
    print(f"\n  High lag-1 ACF: observations are correlated with neighbors.")
else:
    print(f"\n  Low lag-1 ACF: yearly birth weight is largely independent year-to-year.")

# ── Prediction ────────────────────────────────────────────────────────────────

# Bootstrap uncertainty: resample years, refit trend
n_boot = 2000
boot_slopes = []
for _ in range(n_boot):
    idx = np.random.choice(len(years), size=len(years), replace=True)
    _, b = least_squares(years_centered[idx], weights[idx])
    boot_slopes.append(b)

boot_slopes = np.array(boot_slopes)
slope_se = boot_slopes.std()

# Predict for 2005
future_year = 2005
future_centered = future_year - year_mean
pred_2005 = alpha + beta * future_centered
pred_lo   = alpha + (beta - 2 * slope_se) * future_centered
pred_hi   = alpha + (beta + 2 * slope_se) * future_centered

print(f"\n── Prediction for 2005 ─────────────────────────────────────────────────")
print(f"  Point estimate : {pred_2005:.3f} lbs")
print(f"  95% range      : [{pred_lo:.3f}, {pred_hi:.3f}] lbs")
print(f"  (uncertainty from bootstrap of trend slope)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Time series + trend + moving average
ax = axes[0, 0]
ax.plot(years, weights, 'o-', color=COLORS['series'],
        linewidth=2, markersize=6, label='Yearly mean')
trend_line = alpha + beta * (years - year_mean)
ax.plot(years, trend_line, color=COLORS['trend'],
        linewidth=2, linestyle='--', label=f'Trend: {beta:+.3f} lbs/yr')
if len(ma_years) > 1:
    ax.plot(ma_years, ma_vals, color=COLORS['ma'],
            linewidth=2.5, label='3-yr moving avg')
ax.set_xlabel('Year')
ax.set_ylabel('Mean birth weight (lbs)')
ax.set_title('Birth Weight Over Time')
ax.legend(fontsize=9)

# 2. Residuals from trend
ax = axes[0, 1]
trend_resid = weights - trend_line
ax.bar(years, trend_resid, color=[COLORS['trend'] if r > 0 else COLORS['series'] for r in trend_resid],
       alpha=0.8)
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Year')
ax.set_ylabel('Residual from trend (lbs)')
ax.set_title('Detrended Birth Weight')

# 3. ACF plot
ax = axes[1, 0]
lags = np.arange(len(acf_vals))
ax.bar(lags, acf_vals, color=COLORS['series'], alpha=0.8)
ax.axhline(0, color='black', linewidth=1)
# 95% confidence bounds (approximate: ±1.96/sqrt(n))
bound = 1.96 / np.sqrt(len(weights))
ax.axhline(+bound, color='grey', linestyle='--', linewidth=1, label='95% CI bounds')
ax.axhline(-bound, color='grey', linestyle='--', linewidth=1)
ax.set_xlabel('Lag (years)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation Function (ACF)')
ax.legend(fontsize=9)

# 4. Prediction with uncertainty
ax = axes[1, 1]
ax.plot(years, weights, 'o-', color=COLORS['series'], linewidth=2, label='Observed')
x_full = np.linspace(years.min(), 2006, 50)
ax.plot(x_full, alpha + beta * (x_full - year_mean),
        color=COLORS['trend'], linewidth=2, linestyle='--', label='Trend')
ax.scatter([future_year], [pred_2005], color=COLORS['predict'],
           s=100, zorder=5, label=f'Pred 2005: {pred_2005:.3f} lbs')
ax.errorbar([future_year], [pred_2005],
            yerr=[[pred_2005 - pred_lo], [pred_hi - pred_2005]],
            color=COLORS['predict'], linewidth=2, capsize=6)
ax.set_xlabel('Year')
ax.set_ylabel('Mean birth weight (lbs)')
ax.set_title('Trend Extrapolation to 2005')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('ch12_time_series.png', dpi=150)
plt.show()
print("\nFigure saved: ch12_time_series.png")
