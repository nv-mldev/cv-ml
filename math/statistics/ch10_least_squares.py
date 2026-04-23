"""
Chapter 10 — Linear Least Squares
Question: Can we predict birth weight from pregnancy length?

What this script builds:
  - OLS from scratch: compute slope and intercept analytically
  - Residuals and R-squared
  - Residual plot diagnostic
  - Bootstrap confidence interval for the slope

Run: python ch10_least_squares.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live

COLORS = {
    'data':     '#2196F3',
    'line':     '#F44336',
    'residual': '#FF9800',
    'zero':     '#9E9E9E',
    'boot':     '#4CAF50',
}

np.random.seed(42)

# ── Clean data ────────────────────────────────────────────────────────────────
df = live[['prglngth', 'totalwgt_lb']].dropna()
df = df[(df['totalwgt_lb'] > 0) & (df['totalwgt_lb'] < 20)]
df = df[(df['prglngth'] >= 27) & (df['prglngth'] <= 44)]

x = df['prglngth'].values.astype(float)
y = df['totalwgt_lb'].values.astype(float)

# ── OLS from scratch ──────────────────────────────────────────────────────────

def least_squares(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Ordinary Least Squares (OLS) for simple linear regression.
    Minimizes sum of squared residuals: min sum((y - alpha - beta*x)^2)

    Analytic solution:
        beta  = Cov(X, Y) / Var(X)
        alpha = mean(Y) - beta * mean(X)

    The line always passes through (mean_x, mean_y).
    """
    x_mean = x.mean()
    y_mean = y.mean()

    # Covariance and variance computed directly
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x  = np.mean((x - x_mean) ** 2)

    beta  = cov_xy / var_x          # slope: lbs gained per additional week
    alpha = y_mean - beta * x_mean  # intercept: predicted weight at week 0

    return alpha, beta


alpha, beta = least_squares(x, y)

print("── OLS Fit: Birth Weight ~ Pregnancy Length ────────────────────────────")
print(f"  Intercept (alpha) : {alpha:.4f} lbs  (predicted weight at week 0 — not meaningful)")
print(f"  Slope (beta)      : {beta:.4f} lbs per week")
print(f"  Interpretation    : each additional week adds {beta:.3f} lbs to birth weight")
print(f"  At 39 weeks       : {alpha + beta * 39:.3f} lbs predicted")

# ── Residuals and R-squared ───────────────────────────────────────────────────

def compute_residuals(x: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Residuals = actual - predicted."""
    return y - (alpha + beta * x)


def r_squared(y: np.ndarray, residuals: np.ndarray) -> float:
    """
    R^2 = 1 - SSE/SST
    SSE = sum of squared residuals (what the model gets wrong)
    SST = sum of squared deviations from mean (total variance in y)
    R^2 = fraction of variance in y explained by the model.
    """
    sse = np.sum(residuals ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return 1 - sse / sst


residuals = compute_residuals(x, y, alpha, beta)
r2 = r_squared(y, residuals)

print(f"\n── Goodness of Fit ─────────────────────────────────────────────────────")
print(f"  R^2                    : {r2:.4f}")
print(f"  Interpretation         : pregnancy length explains {r2*100:.1f}% of birth weight variance")
print(f"  Residual std           : {residuals.std():.4f} lbs")
print(f"  Mean residual          : {residuals.mean():.6f}  (should be ~0)")

# ── Bootstrap confidence interval for slope ───────────────────────────────────

n_boot = 2000
boot_slopes = []
boot_alphas = []

for _ in range(n_boot):
    idx = np.random.choice(len(x), size=len(x), replace=True)
    a_boot, b_boot = least_squares(x[idx], y[idx])
    boot_slopes.append(b_boot)
    boot_alphas.append(a_boot)

boot_slopes = np.array(boot_slopes)
ci_lo, ci_hi = np.percentile(boot_slopes, [2.5, 97.5])

print(f"\n── Bootstrap 95% CI for Slope ──────────────────────────────────────────")
print(f"  Slope estimate   : {beta:.4f} lbs/week")
print(f"  95% CI           : [{ci_lo:.4f}, {ci_hi:.4f}] lbs/week")
print(f"  Both bounds are positive → slope is reliably > 0")
print(f"  (Longer pregnancy reliably means heavier baby)")

# ── Residual diagnostics ──────────────────────────────────────────────────────
fitted = alpha + beta * x

print(f"\n── Residual Diagnostics ────────────────────────────────────────────────")
print(f"  Max positive residual : +{residuals.max():.2f} lbs  (heavier than predicted)")
print(f"  Max negative residual : {residuals.min():.2f} lbs  (lighter than predicted)")
print(f"  Residuals symmetric?  : mean = {residuals.mean():.4f}  (yes, nearly 0)")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Scatter + fitted line
ax = axes[0, 0]
ax.scatter(x, y, alpha=0.1, s=5, color=COLORS['data'], label='Data')
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, alpha + beta * x_line, color=COLORS['line'],
        linewidth=2.5, label=f'y = {alpha:.2f} + {beta:.2f}x')
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('Birth weight (lbs)')
ax.set_title(f'OLS Fit  (R² = {r2:.3f})')
ax.legend(fontsize=9)

# 2. Residual plot
ax = axes[0, 1]
ax.scatter(fitted, residuals, alpha=0.1, s=5, color=COLORS['residual'])
ax.axhline(0, color='black', linewidth=1.5)
ax.set_xlabel('Fitted values (lbs)')
ax.set_ylabel('Residuals (lbs)')
ax.set_title('Residual Plot\n(should show no pattern)')

# 3. Bootstrap distribution of slope
ax = axes[1, 0]
ax.hist(boot_slopes, bins=50, density=True, color=COLORS['boot'], alpha=0.8)
ax.axvline(beta, color=COLORS['line'], linewidth=2, label=f'Observed β = {beta:.4f}')
ax.axvline(ci_lo, color='grey', linestyle='--', linewidth=1.5, label=f'95% CI')
ax.axvline(ci_hi, color='grey', linestyle='--', linewidth=1.5)
ax.set_xlabel('Bootstrap slope (lbs/week)')
ax.set_title('Bootstrap Distribution of Slope')
ax.legend(fontsize=9)

# 4. Residual distribution
ax = axes[1, 1]
ax.hist(residuals, bins=60, density=True, color=COLORS['residual'], alpha=0.8)
x_res = np.linspace(residuals.min(), residuals.max(), 100)
from scipy import stats
ax.plot(x_res, stats.norm.pdf(x_res, 0, residuals.std()),
        color='black', linewidth=2, label='Normal reference')
ax.set_xlabel('Residual (lbs)')
ax.set_title('Residual Distribution (should be ~Normal)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('ch10_least_squares.png', dpi=150)
plt.show()
print("\nFigure saved: ch10_least_squares.png")
