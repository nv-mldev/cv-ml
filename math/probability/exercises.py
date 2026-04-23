"""
Exercises: 01a Probability for Computer Vision

Stub file for the four exercises in this module. Each function has a docstring
explaining what to implement and what the expected output should look like.

Run after completing: python exercises.py

Exercises:
  1. Binomial to Poisson convergence — quantitative O(1/n) rate
  2. Anscombe variance-stabilising transform for Poisson data
  3. Build a real-world noise budget for a Basler industrial camera
  4. CLT convergence rate vs. distribution skewness
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, kstest

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}

np.random.seed(42)


# ── Exercise 1 ──────────────────────────────────────────────────────────────

def exercise1_binomial_poisson_convergence() -> None:
    """Quantify the convergence rate of Binomial(n, λ/n) → Poisson(λ).

    Task:
        For λ = 50, compute the maximum absolute difference between
        Binomial(n, λ/n) and Poisson(λ) PMFs for n ∈ {50, 100, 500, 1000, 5000}.
        Plot the max difference vs n on a log-log scale and estimate the slope.

    Steps:
        1. For each n, compute binom.pmf(k, n, λ/n) and poisson.pmf(k, λ)
           over k ∈ [0, 100]
        2. max_diff[n] = max |binom_pmf - poisson_pmf|
        3. Plot max_diff vs n on log-log scale
        4. Fit np.polyfit(log(n), log(max_diff), 1) to estimate slope
           (slope ≈ -1 means O(1/n) convergence)

    Expected output:
        A log-log plot showing max |difference| decreasing approximately as O(1/n).
        Slope printed to console should be close to -1.0.
    """
    # YOUR CODE HERE
    pass


# ── Exercise 2 ──────────────────────────────────────────────────────────────

def exercise2_anscombe_transform() -> None:
    """Demonstrate the Anscombe variance-stabilising transform for Poisson data.

    The Anscombe transform f(x) = 2√(x + 3/8) converts Poisson(λ) data
    (where variance = λ) into approximately unit-variance data — enabling
    standard Gaussian denoisers (NLM, BM3D, Wiener) on Poisson-distributed images.

    Task:
        1. Generate 10,000 Poisson samples for λ ∈ {5, 50, 500}
        2. Apply the Anscombe transform: f(x) = 2 * sqrt(x + 3/8)
        3. Compute variance before and after the transform for each λ
        4. Plot raw samples histogram vs transformed histogram side by side

    Steps:
        1. raw_samples = np.random.poisson(lam, size=10000)
        2. transformed = 2 * np.sqrt(raw_samples + 3/8)
        3. Print: f"λ={lam}: raw var={raw_samples.var():.1f}, transformed var={transformed.var():.3f}"
        4. Plot histograms: before (left) and after (right) for all three λ values

    Expected output:
        Raw variance grows with λ (5, 50, 500).
        Transformed variance is approximately 1.0 for all three λ values.
    """
    # YOUR CODE HERE
    pass


# ── Exercise 3 ──────────────────────────────────────────────────────────────

def exercise3_noise_budget() -> None:
    """Build a complete noise budget for a Basler acA1920-40gm industrial camera.

    Sensor specs:
        - Scene illumination: 200 photons/µm²/exposure
        - Photosite pitch: 3.45 µm
        - Quantum efficiency: 0.65
        - Read noise: 4 electrons
        - Dark current: 2 electrons (cooled sensor)
        - Full well: 11,000 electrons
        - Bit depth: 12

    Task:
        1. Compute expected_electrons = flux × pitch² × QE
        2. Noise budget: shot_var = expected_electrons,
                         read_var = read_noise²,
                         dark_var = dark_current
                         total_noise = sqrt(shot + read + dark)
        3. Compute SNR = expected_electrons / total_noise
        4. Find the signal level where shot noise > read noise
           (read-noise limited → shot-noise limited crossover)
        5. Simulate 10,000 exposures using the full signal chain and
           overlay the histogram against the theoretical Normal(μ, σ²)

    Steps:
        - Use np.random.poisson for shot noise, np.random.normal for read noise,
          np.random.poisson for dark current
        - Print a noise budget table: source | variance | std dev | % of total
        - Plot: simulated histogram + theoretical Normal overlay

    Expected output:
        Noise budget table printed to console.
        Histogram matching theoretical predictions (simulation ≈ theory).
        SNR printed for this illumination level.
    """
    # YOUR CODE HERE
    pass


# ── Exercise 4 ──────────────────────────────────────────────────────────────

def exercise4_clt_convergence_skewness() -> None:
    """Measure CLT convergence rate as a function of distribution skewness.

    Distributions (in order of increasing skewness):
        - Uniform(0, 1)    — skewness = 0 (symmetric)
        - Exponential(1)   — skewness = 2
        - Pareto(2, 1)     — skewness > 2 (heavy-tailed, very skewed)

    Task:
        1. For each distribution and each n in [1, 2, 5, 10, 30, 100]:
           - Draw 10,000 normalized sums of n variables
           - Run kstest(standardised, 'norm') → KS statistic
        2. Plot KS distance vs n for all three distributions on log-log axes
        3. Rank by convergence speed and print the ranking

    Steps:
        - Pareto samples: np.random.pareto(2, size) + 1  (shape=2 for finite variance)
        - Normalize sums: (total - total.mean()) / total.std()
        - ks_stat, _ = kstest(normalised, 'norm')
        - Plot all three curves on the same axes

    Expected output:
        Log-log plot showing Pareto converges slowest, Uniform converges fastest.
        Console ranking: Uniform < Exponential < Pareto (ascending KS distance).
    """
    # YOUR CODE HERE
    pass


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Exercise 1: Binomial → Poisson Convergence Rate")
    print("=" * 60)
    exercise1_binomial_poisson_convergence()

    print()
    print("=" * 60)
    print("Exercise 2: Anscombe Variance-Stabilising Transform")
    print("=" * 60)
    exercise2_anscombe_transform()

    print()
    print("=" * 60)
    print("Exercise 3: Noise Budget for Industrial Camera")
    print("=" * 60)
    exercise3_noise_budget()

    print()
    print("=" * 60)
    print("Exercise 4: CLT Convergence vs Distribution Skewness")
    print("=" * 60)
    exercise4_clt_convergence_skewness()
