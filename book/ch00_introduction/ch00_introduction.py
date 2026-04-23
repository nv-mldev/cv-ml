"""
Chapter 0 — From Measurements to Meaning

Runnable simulations for the introductory chapter. Produces four figures
referenced by ch00_introduction.md:

  - ch00_clean_world.png      — ideal sensor, no noise
  - ch00_repeated_readings.png — 1000 readings at fixed x, histogram
  - ch00_clt.png               — sum of uniforms converging to Gaussian
  - ch00_three_attacks.png     — averaging / linear fit / poly fit on same data

Run:
    python book/ch00_introduction/ch00_introduction.py

Outputs are written next to this script.
"""

# --- Setup ---
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    'primary':   '#2196F3',   # blue
    'secondary': '#4CAF50',   # green
    'result':    '#FFC107',   # amber
    'highlight': '#F44336',   # red
    'transform': '#9C27B0',   # purple
    'gradient':  '#FF9800',   # orange
}

SCRIPT_DIR = Path(__file__).parent

np.random.seed(42)
print("Setup complete.")


# ── Figure 1: The clean world ──────────────────────────────────────────
# Plot the ideal, noise-free f(x) = 2x + 0.5 as a calibration curve.
# Purpose: anchor the reader's mental image of what a perfect sensor would
# give you, before we break that picture in Figure 2.
# ───────────────────────────────────────────────────────────────────────

x_grid = np.linspace(0, 10, 200)
true_f = lambda xv: 2.0 * xv + 0.5
y_clean = true_f(x_grid)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x_grid, y_clean, color=COLORS['secondary'], linewidth=2.5,
        label='y = f(x) (the truth)')
ax.set_xlabel('x (knob setting / input)')
ax.set_ylabel('y (sensor reading)')
ax.set_title('The clean world — perfect sensor, no ML needed')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'ch00_clean_world.png', dpi=140)
plt.close(fig)
print("Wrote ch00_clean_world.png")


# ── Figure 2: Repeated readings at a fixed x ───────────────────────────
# Take 1000 independent noisy measurements at x = 5 and show:
#   (a) the sequence of readings (nothing is constant),
#   (b) their distribution (a clean bell shape).
# This is the moment the reader sees that noise has STRUCTURE — it is
# unpredictable per sample but predictable in aggregate.
# ───────────────────────────────────────────────────────────────────────

x_fixed = 5.0
true_value = true_f(x_fixed)
num_repeats = 1000
noise_std = 0.8

# Each reading = true value plus an independent Gaussian perturbation.
repeated_readings = true_value + np.random.randn(num_repeats) * noise_std

print(f"True value at x = {x_fixed}: {true_value}")
print(f"Sample mean of {num_repeats} readings: {repeated_readings.mean():.4f}")
print(f"Sample std of {num_repeats} readings:  {repeated_readings.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: every reading as a dot
axes[0].scatter(range(num_repeats), repeated_readings, s=8,
                color=COLORS['primary'], alpha=0.6)
axes[0].axhline(true_value, color=COLORS['secondary'], linewidth=2,
                linestyle='--', label=f'true value = {true_value}')
axes[0].set_xlabel('reading number')
axes[0].set_ylabel('sensor output y')
axes[0].set_title(f'{num_repeats} readings at the same x = {x_fixed}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: histogram of those same readings
axes[1].hist(repeated_readings, bins=40, color=COLORS['primary'],
             alpha=0.7, edgecolor='white')
axes[1].axvline(true_value, color=COLORS['secondary'], linewidth=2,
                linestyle='--', label=f'true value = {true_value}')
axes[1].axvline(repeated_readings.mean(), color=COLORS['highlight'],
                linewidth=2,
                label=f'sample mean = {repeated_readings.mean():.3f}')
axes[1].set_xlabel('sensor output y')
axes[1].set_ylabel('count')
axes[1].set_title('Distribution of readings — the noise has structure')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'ch00_repeated_readings.png', dpi=140)
plt.close(fig)
print("Wrote ch00_repeated_readings.png")


# ── Figure 3: Central Limit Theorem in action ──────────────────────────
# Build "noise" as the sum of K uniform(-0.5, 0.5) samples. The individual
# uniform is decidedly non-Gaussian, but as K grows, the sum converges to
# a Gaussian. Overlay the matched-variance Gaussian PDF to show the fit.
#
# Variance of uniform(-0.5, 0.5) = 1/12, so sum of K has variance K/12 and
# std deviation sqrt(K/12).
# ───────────────────────────────────────────────────────────────────────

num_draws = 50000
K_values = [1, 3, 10, 30]

fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))

for ax, K in zip(axes, K_values):
    # Draw num_draws independent sums of K uniforms in [-0.5, 0.5].
    uniform_samples = np.random.uniform(-0.5, 0.5, size=(num_draws, K))
    sums = uniform_samples.sum(axis=1)

    ax.hist(sums, bins=60, density=True, color=COLORS['primary'],
            alpha=0.7, edgecolor='white')

    # Gaussian with matched mean (0) and variance (K/12).
    sigma = np.sqrt(K / 12.0)
    x_plot = np.linspace(sums.min(), sums.max(), 200)
    gaussian_pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-x_plot ** 2 / (2 * sigma ** 2)))
    ax.plot(x_plot, gaussian_pdf, color=COLORS['highlight'], linewidth=2,
            label=f'Gaussian σ={sigma:.2f}')

    ax.set_title(f'sum of {K} uniform(s)')
    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Central limit theorem — summing many small independent noises '
             '→ Gaussian',
             fontsize=13)
fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'ch00_clt.png', dpi=140)
plt.close(fig)
print("Wrote ch00_clt.png")


# ── Figure 4: Three attacks on the same data ───────────────────────────
# Generate a mildly nonlinear true f, sample it noisily, then solve the
# same inverse problem three different ways:
#   Attack 1: average 100 readings at x = 3 (sharp point estimate)
#   Attack 2: straight-line least-squares fit (global but underfits wiggle)
#   Attack 3: degree-10 polynomial fit (tracks wiggle, risks overfitting)
# Show all three on one figure so the reader sees each attack's tradeoff.
# ───────────────────────────────────────────────────────────────────────

np.random.seed(3)
true_f_wavy = lambda xv: 1.0 + 0.5 * xv + 1.2 * np.sin(1.5 * xv)

n_points = 60
x_samples = np.sort(np.random.uniform(0, 6, n_points))
y_samples = true_f_wavy(x_samples) + np.random.randn(n_points) * 0.4

x_fine = np.linspace(0, 6, 400)
y_fine_true = true_f_wavy(x_fine)

# --- Attack 1: averaging at x = 3 ---
# We pretend we can re-measure many times at the same x.
x_query = 3.0
true_at_query = true_f_wavy(x_query)
repeats_at_query = true_at_query + np.random.randn(100) * 0.4
averaged_estimate = repeats_at_query.mean()
averaged_sem = repeats_at_query.std() / np.sqrt(len(repeats_at_query))

# --- Attack 2: straight-line least-squares fit ---
# Closed-form slope/intercept from the scalar formulas.
x_bar = x_samples.mean()
y_bar = y_samples.mean()
slope_hat = (np.sum((x_samples - x_bar) * (y_samples - y_bar))
             / np.sum((x_samples - x_bar) ** 2))
intercept_hat = y_bar - slope_hat * x_bar
y_fit_linear = slope_hat * x_fine + intercept_hat

# --- Attack 3: degree-10 polynomial fit via lstsq ---
# Build a Vandermonde-style design matrix with columns x^0 ... x^10.
degree = 10
def poly_features(x, deg):
    return np.column_stack([x ** p for p in range(deg + 1)])

phi_train = poly_features(x_samples, degree)
phi_fine  = poly_features(x_fine, degree)
weights_poly, *_ = np.linalg.lstsq(phi_train, y_samples, rcond=None)
y_fit_poly = phi_fine @ weights_poly

# --- Combined plot ---
fig, ax = plt.subplots(figsize=(11, 6))

ax.scatter(x_samples, y_samples, color='black', s=30, alpha=0.7,
           label='noisy measurements', zorder=3)
ax.plot(x_fine, y_fine_true, color=COLORS['secondary'], linewidth=2.5,
        linestyle='--', label='true f(x)   [usually unknown]')

# Attack 1: single point estimate with errorbar
ax.errorbar(x_query, averaged_estimate, yerr=averaged_sem,
            fmt='o', color=COLORS['highlight'], markersize=12, capsize=8,
            label=f'Attack 1: average at x={x_query}', zorder=5)

# Attack 2
ax.plot(x_fine, y_fit_linear, color=COLORS['primary'], linewidth=2.5,
        label='Attack 2: linear fit')

# Attack 3
ax.plot(x_fine, y_fit_poly, color=COLORS['transform'], linewidth=2.5,
        label=f'Attack 3: degree-{degree} polynomial fit')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Three attacks on the same noisy dataset')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(SCRIPT_DIR / 'ch00_three_attacks.png', dpi=140)
plt.close(fig)
print("Wrote ch00_three_attacks.png")

print("\nAll figures generated.")
