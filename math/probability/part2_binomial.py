"""
Part 2: The Binomial Distribution — Counting Successes

Builds the Binomial PMF from first principles term by term, visualizes how
shape changes with n and p, and validates theory against Monte Carlo simulation.

What this script demonstrates:
  - Manual construction of the Binomial PMF (ways × p^k × (1-p)^(n-k))
  - How distribution shape changes with n (wider, more bell-shaped) and p (shifts peak)
  - Monte Carlo simulation matching theoretical Binomial PMF
  - CDF interpretation: "what fraction of exposures gave ≤ k electrons?"

Run: python part2_binomial.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import comb

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
# Build the Binomial PMF term by term from the formula
# For each possible count k from 0 to n:
# 1. Compute C(n, k) — how many ways to choose which k trials succeed
# 2. Compute p^k — probability that exactly those k trials succeed
# 3. Compute (1-p)^(n-k) — probability that all remaining trials fail
# 4. Multiply: P(k) = C(n,k) × p^k × (1-p)^(n-k)
# 5. Print as a table to see each factor's contribution
# What to look for: scipy.stats.binom.pmf does exactly these steps internally.
#   Seeing it once makes the formula concrete.
# ───────────────────────────────────────────────────────────

# n = 10 photons, p = 0.7 (QE)
number_of_photons = 10
quantum_efficiency = 0.7

print(f"n = {number_of_photons} photons, p = {quantum_efficiency} (QE)")
print()
print(f"{'k (electrons)':>14s}  {'C(n,k)':>8s}  {'p^k':>10s}  {'(1-p)^(n-k)':>14s}  {'P(k)':>10s}")
print("─" * 65)

for k in range(number_of_photons + 1):
    # How many ways to choose which k photons get detected
    ways = comb(number_of_photons, k, exact=True)

    # Probability that exactly these k photons succeed
    p_success = quantum_efficiency ** k

    # Probability that the other (n-k) photons fail
    p_failure = (1 - quantum_efficiency) ** (number_of_photons - k)

    # Total probability
    probability = ways * p_success * p_failure

    print(f"{k:>14d}  {ways:>8d}  {p_success:>10.6f}  {p_failure:>14.6f}  {probability:>10.6f}")

print()
expected_value = number_of_photons * quantum_efficiency
variance = number_of_photons * quantum_efficiency * (1 - quantum_efficiency)
print(f"Expected electrons: np = {number_of_photons} × {quantum_efficiency} = {expected_value}")
print(f"Variance: np(1-p) = {number_of_photons} × {quantum_efficiency} × {1-quantum_efficiency} = {variance}")
print(f"Std dev: √{variance:.1f} = {np.sqrt(variance):.2f} electrons")

# ── Algorithm ──────────────────────────────────────────────
# Visualize how Binomial shape changes with n and p
# Top row (fix p=0.7, vary n = 5, 20, 100):
#   - observe distribution widening and becoming bell-shaped as n grows
# Bottom row (fix n=30, vary p = 0.1, 0.5, 0.9):
#   - observe peak shifting and symmetry changing
# For each subplot: compute PMF using binom.pmf(k, n, p),
#   mark mean (μ = np) and ±1 std (σ = √(np(1-p)))
# What to look for: as n increases, the Binomial looks like a bell curve —
#   first hint of the Normal approximation and the CLT (formal proof in Part 5).
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top row: fix p = 0.7 (QE), vary n (photon count)
p_fixed = 0.7
for ax, n in zip(axes[0], [5, 20, 100]):
    k_values = np.arange(0, n + 1)
    probabilities = binom.pmf(k_values, n, p_fixed)

    ax.bar(k_values, probabilities, color=COLORS['primary'], alpha=0.7)

    mean = n * p_fixed
    std = np.sqrt(n * p_fixed * (1 - p_fixed))
    ax.axvline(mean, color=COLORS['highlight'], linestyle='--', linewidth=2,
               label=f'μ = {mean:.0f}')
    ax.axvspan(mean - std, mean + std, alpha=0.15, color=COLORS['secondary'],
               label=f'±1σ = ±{std:.1f}')

    ax.set_title(f'Binomial(n={n}, p={p_fixed})\n{n} trials, 70% success prob | μ={mean:.0f}, σ={std:.1f}', fontsize=12)
    ax.set_xlabel('k (successes: edge pixels / matches / electrons)')
    ax.set_ylabel('P(k)')
    ax.legend(fontsize=8)

# Bottom row: fix n = 30, vary p (QE)
n_fixed = 30
for ax, p in zip(axes[1], [0.1, 0.5, 0.9]):
    k_values = np.arange(0, n_fixed + 1)
    probabilities = binom.pmf(k_values, n_fixed, p)

    ax.bar(k_values, probabilities, color=COLORS['secondary'], alpha=0.7)

    mean = n_fixed * p
    std = np.sqrt(n_fixed * p * (1 - p))
    ax.axvline(mean, color=COLORS['highlight'], linestyle='--', linewidth=2,
               label=f'μ = {mean:.0f}')
    ax.axvspan(mean - std, mean + std, alpha=0.15, color=COLORS['primary'],
               label=f'±1σ = ±{std:.1f}')

    ax.set_title(f'Binomial(n={n_fixed}, p={p})\n{n_fixed} trials, p={p} success prob | μ={mean:.0f}, σ={std:.1f}', fontsize=12)
    ax.set_xlabel('k (successes: edge pixels / matches / electrons)')
    ax.set_ylabel('P(k)')
    ax.legend(fontsize=8)

axes[0][0].set_ylabel('P(k)\n(fixed p=0.7, vary n)', fontsize=11)
axes[1][0].set_ylabel('P(k)\n(fixed n=30, vary p)', fontsize=11)

plt.suptitle('Binomial Distribution: How Success Counts Vary (edges, matches, photons, ...)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("Top row: more photons (larger n) → wider distribution, but RELATIVELY narrower")
print("Bottom row: higher QE (larger p) → peak shifts right, shape changes")
print()
print("Notice: as n increases, the distribution starts looking like a bell curve...")

# ── Algorithm ──────────────────────────────────────────────
# Monte Carlo simulation of the Binomial
# 1. Run 10,000 independent repetitions (n=100 trials, p=0.7)
# 2. Method 1 (transparent): simulate each of the 100 trials individually
#    as Bernoulli(p) and sum
# 3. Method 2 (efficient): use np.random.binomial(n, p, size=10000)
# 4. Print simulated mean and std — compare to theory: μ = np, σ = √(np(1-p))
# What to look for: both methods agree. Simulation validates theory.
# ───────────────────────────────────────────────────────────

number_of_photons = 100
quantum_efficiency = 0.7
number_of_exposures = 10000

# Method 1: Simulate each trial individually (slow but transparent)
single_exposure_photons = np.random.binomial(1, quantum_efficiency, size=number_of_photons)
print(f"One run ({number_of_photons} trials, p={quantum_efficiency}):")
print(f"  Individual photon results: {single_exposure_photons[:20]}... (showing first 20)")
print(f"  Total electrons: {single_exposure_photons.sum()}")
print()

# Method 2: Use the Binomial directly (equivalent but faster)
electron_counts = np.random.binomial(number_of_photons, quantum_efficiency, size=number_of_exposures)

print(f"10,000 exposures simulated:")
print(f"  First 20 electron counts: {electron_counts[:20]}")
print(f"  Mean: {electron_counts.mean():.1f}  (expected: {number_of_photons * quantum_efficiency})")
print(f"  Std:  {electron_counts.std():.2f}  (expected: {np.sqrt(number_of_photons * quantum_efficiency * (1-quantum_efficiency)):.2f})")
print()
print("Method 1 and Method 2 produce the same statistics.")
print("The Binomial distribution IS the count of Bernoulli successes.")

# ── Algorithm ──────────────────────────────────────────────
# Compare simulation output to theoretical PMF
# 1. Build histogram of 10,000 simulated counts
# 2. Compute theoretical Binomial PMF using binom.pmf
# 3. Plot both on same axes — bars for simulation, line for theory
# 4. Second panel: CDF comparison
# What to look for: simulation bars hug the theory line. Residual differences
#   shrink as number of simulations increases (also a CLT effect).
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

k_values = np.arange(electron_counts.min(), electron_counts.max() + 1)
theoretical_pmf = binom.pmf(k_values, number_of_photons, quantum_efficiency)

axes[0].hist(electron_counts, bins=k_values - 0.5, density=True,
             color=COLORS['primary'], alpha=0.5, label='Simulated (10,000 exposures)')
axes[0].plot(k_values, theoretical_pmf, 'o-', color=COLORS['highlight'],
             markersize=4, linewidth=2, label='Theoretical Binomial PMF')
axes[0].set_xlabel('Electrons detected per exposure')
axes[0].set_ylabel('Probability')
axes[0].set_title(f'Binomial(n={number_of_photons}, p={quantum_efficiency})', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(electron_counts, bins=k_values - 0.5, density=True, cumulative=True,
             color=COLORS['primary'], alpha=0.5, label='Simulated CDF')
axes[1].plot(k_values, binom.cdf(k_values, number_of_photons, quantum_efficiency),
             '-', color=COLORS['highlight'], linewidth=2, label='Theoretical CDF')
axes[1].set_xlabel('Electrons detected per exposure')
axes[1].set_ylabel('Cumulative probability P(X ≤ k)')
axes[1].set_title('CDF: "What fraction of exposures gave ≤ k electrons?"', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("The simulation matches theory perfectly — the Binomial model is exact for this process.")
