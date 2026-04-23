"""
Part 1: The Bernoulli Trial — The Atom of Randomness

Simulates a single Bernoulli experiment to build intuition for the fundamental
binary event that underlies all other distributions in this series.

What this script demonstrates:
  - A Bernoulli trial as a binary outcome (0 or 1)
  - The quantum efficiency (QE) of a camera sensor as a Bernoulli probability
  - The law of large numbers in miniature: total successes hover near n × p

Run: python part1_bernoulli.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt

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
# Simulate a single Bernoulli experiment
# 1. Set p = 0.7 — probability of success per trial
#    (QE, pixel-exceeds-threshold probability, or match probability)
# 2. Run n = 20 independent trials using np.random.binomial(1, p, size=n)
# 3. Print the outcome array — each element is 0 (failure) or 1 (success)
# 4. Print total successes and compare to expected n × p = 14
# What to look for: outcomes are random, but total count hovers near n × p.
#   Run this conceptually multiple times — the sequence changes but the average
#   stays the same. This is the law of large numbers in miniature.
# ───────────────────────────────────────────────────────────

quantum_efficiency = 0.7  # or: probability a pixel exceeds threshold, or match probability
number_of_trials = 20     # could be: 20 pixels tested, 20 descriptor pairs compared, 20 photons

# Each trial: did it succeed? (1 = yes, 0 = no)
trial_outcomes = np.random.binomial(1, quantum_efficiency, size=number_of_trials)

print(f"Success probability (p): {quantum_efficiency}  (QE, threshold probability, or match probability)")
print(f"Number of trials: {number_of_trials}")
print()
print(f"Each trial's result: {trial_outcomes}")
print(f"  1 = success (electron produced / pixel above threshold / descriptor matched)")
print(f"  0 = failure")
print()
print(f"Total successes: {trial_outcomes.sum()} out of {number_of_trials} trials")
print(f"Success rate this run: {trial_outcomes.sum()/number_of_trials:.0%}")
print(f"Expected successes (p × n): {quantum_efficiency * number_of_trials:.1f}")
