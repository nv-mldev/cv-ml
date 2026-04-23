"""
Part 3: The Poisson Distribution — The Limit of Many Rare Events

Visualizes the Binomial-to-Poisson convergence as n → ∞ (with np = λ fixed),
and demonstrates Poisson shot noise across a sensor gradient covering all
three noise regimes.

What this script demonstrates:
  - Binomial(n, λ/n) converges to Poisson(λ) as n grows — max diff → 0
  - Poisson noise along a brightness gradient: dark pixels noisy, bright cleaner
  - σ = √λ property verified against 1,000 simulated captures
  - SNR = λ/√λ = √λ: the fundamental sensor quality metric

Run: python part3_poisson.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}

np.random.seed(42)
print("Setup complete.")

# ── Algorithm ──────────────────────────────────────────────
# Watch Binomial converge to Poisson as n → ∞
# 1. Fix λ = 20 (expected count)
# 2. For each n in [20, 50, 100, 500, 2000, 10000]:
#    - Set p = λ/n so that np = λ stays constant while p shrinks
#    - Compute Binomial(n, p) PMF
#    - Compute Poisson(λ) PMF over the same range
#    - Plot both side by side and show max |diff|
# What to look for: max difference decreases by roughly 1/n.
#   The Poisson is not a separate distribution — it IS the limit of Binomial.
# ───────────────────────────────────────────────────────────

lambda_fixed = 20  # expected photons per exposure

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
n_values = [20, 50, 100, 500, 2000, 10000]

for ax, n in zip(axes.flat, n_values):
    p = lambda_fixed / n  # p shrinks as n grows, keeping λ = np constant

    k_range = np.arange(max(0, int(lambda_fixed - 4*np.sqrt(lambda_fixed))),
                        int(lambda_fixed + 4*np.sqrt(lambda_fixed)) + 1)
    binom_pmf = binom.pmf(k_range, n, p)
    poisson_pmf = poisson.pmf(k_range, lambda_fixed)

    ax.bar(k_range - 0.15, binom_pmf, width=0.3, color=COLORS['primary'],
           alpha=0.7, label=f'Binomial(n={n}, p={p:.4f})')
    ax.bar(k_range + 0.15, poisson_pmf, width=0.3, color=COLORS['highlight'],
           alpha=0.7, label=f'Poisson(λ={lambda_fixed})')

    max_diff = np.max(np.abs(binom_pmf - poisson_pmf))
    ax.set_title(f'n = {n}, p = {p:.5f}\nmax |diff| = {max_diff:.6f}', fontsize=11)
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')
    ax.legend(fontsize=7)

plt.suptitle(f'Binomial → Poisson Convergence (λ = np = {lambda_fixed} fixed)',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"λ = np = {lambda_fixed} is held constant throughout.")
print(f"As n increases from 20 to 10,000:")
print(f"  - p decreases from {lambda_fixed/20:.4f} to {lambda_fixed/10000:.6f}")
print(f"  - The Binomial becomes indistinguishable from the Poisson")
print(f"  - By n ≈ 100, the difference is already negligible")

# ── Algorithm ──────────────────────────────────────────────
# Poisson noise along a gradient — all three noise regimes
# 1. Create 1D scene: 200 pixels with λ increasing linearly from 5 to 200
# 2. True scene: plot λ(x) — the noiseless ground truth
# 3. Noisy measurement: draw one Poisson sample per pixel
# 4. Noise magnitude: compute std from 1,000 captures and compare to √λ
# What to look for: dark pixels are noisiest relative to signal (low SNR);
#   bright pixels have more absolute noise but better SNR. σ = √λ confirmed.
# ───────────────────────────────────────────────────────────

pixel_positions = np.arange(200)
expected_photons = 5 + 195 * (pixel_positions / 199)  # λ ranges from 5 to 200

fig, axes = plt.subplots(3, 1, figsize=(16, 10))

# Row 1: The true scene (no noise)
axes[0].plot(pixel_positions, expected_photons, 'k-', linewidth=2)
axes[0].fill_between(pixel_positions, expected_photons, alpha=0.1, color='black')
axes[0].set_ylabel('Expected photons (λ)', fontsize=12)
axes[0].set_title('True Scene: Smooth Gradient (deterministic — no randomness)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Row 2: One noisy capture (Poisson draw)
one_capture = np.random.poisson(expected_photons.astype(int))
axes[1].bar(pixel_positions, one_capture, width=1, color=COLORS['primary'], alpha=0.7)
axes[1].plot(pixel_positions, expected_photons, 'k--', linewidth=1.5, label='True signal')
axes[1].set_ylabel('Detected photons', fontsize=12)
axes[1].set_title('One Exposure: Poisson Noise (notice more noise on the left = dark region)', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Row 3: Noise level — measured std vs theoretical √λ
many_captures = np.random.poisson(expected_photons.astype(int), size=(1000, 200))
measured_std = many_captures.std(axis=0)
theoretical_std = np.sqrt(expected_photons)

axes[2].plot(pixel_positions, measured_std, color=COLORS['primary'], linewidth=1.5,
             label='Measured std (1000 captures)')
axes[2].plot(pixel_positions, theoretical_std, 'k--', linewidth=2,
             label='Theoretical √λ')
axes[2].set_xlabel('Pixel position', fontsize=12)
axes[2].set_ylabel('Noise (std dev)', fontsize=12)
axes[2].set_title('Noise Increases with Signal: σ = √λ (Poisson property)', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Dark pixels (left): λ ≈ 5, noise ≈ √5 ≈ 2.2 → relative noise = 45%")
print("Bright pixels (right): λ ≈ 200, noise ≈ √200 ≈ 14.1 → relative noise = 7%")
print()
print("The noise GROWS in absolute terms, but SHRINKS relative to the signal.")
print("This is the fundamental property of Poisson noise: σ = √λ, so SNR = √λ.")
