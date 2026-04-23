"""
Part 4: The Normal (Gaussian) Distribution — The Bell Curve

Visualizes how the Normal distribution's two parameters control its shape,
and demonstrates the Poisson-to-Normal convergence as λ grows.

What this script demonstrates:
  - μ shifts the bell left/right without changing shape
  - σ widens or narrows the bell without changing its center
  - Poisson(λ) bars align with Normal(μ=λ, σ²=λ) curve as λ increases
  - At λ = 1: clearly not Gaussian; at λ = 500: indistinguishable

Run: python part4_normal.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm

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
# Visualize the Normal distribution and its two parameters
# Left panel (vary μ, fix σ=2): three Normal PDFs with different means —
#   show μ shifts the center without changing the shape
# Right panel (fix μ=0, vary σ): three Normal PDFs with different σ —
#   show σ controls the width
# For each curve: compute norm.pdf(x, mu, sigma) over a fine x grid
#   and shade under the curve
# What to look for: μ = "where is the signal?" σ = "how noisy is it?"
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

x = np.linspace(-10, 30, 500)

# Left: vary μ (shift the center)
ax = axes[0]
for mu, color in [(0, COLORS['primary']), (5, COLORS['secondary']), (15, COLORS['gradient'])]:
    sigma = 2
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, linewidth=2, color=color, label=f'μ={mu}, σ={sigma}')
    ax.fill_between(x, pdf, alpha=0.1, color=color)
ax.set_title('Varying μ (mean): shifts the bell', fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

# Right: vary σ (change the width)
ax = axes[1]
for sigma, color in [(1, COLORS['primary']), (2, COLORS['secondary']), (4, COLORS['gradient'])]:
    mu = 10
    pdf = norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, linewidth=2, color=color, label=f'μ={mu}, σ={sigma}')
    ax.fill_between(x, pdf, alpha=0.1, color=color)
ax.set_title('Varying σ (std dev): widens the bell', fontsize=12)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("μ controls WHERE the peak is (the 'signal')")
print("σ controls HOW WIDE the spread is (the 'noise')")
print("The area under every curve is exactly 1 (total probability)")

# ── Algorithm ──────────────────────────────────────────────
# Watch Poisson converge to Normal as λ grows
# 1. For each λ in [1, 5, 10, 30, 100, 500]:
#    - Compute Poisson(λ) PMF over k = [λ - 4√λ, λ + 4√λ]
#    - Compute approximating Normal(μ=λ, σ²=λ) PDF over the same range
#    - Plot both on same axes
# What to look for:
#   λ = 1:   Very skewed, clearly NOT Gaussian
#   λ = 5:   Still asymmetric, Gaussian is a rough fit
#   λ = 10:  Getting close — the rule-of-thumb minimum
#   λ = 30:  Good match — most practical sensor scenarios
#   λ = 100: Excellent — typical industrial camera pixel
#   λ = 500: Indistinguishable — why we treat bright pixels as Gaussian
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

lambda_values = [1, 5, 10, 30, 100, 500]

for ax, lam in zip(axes.flat, lambda_values):
    k_range = np.arange(max(0, int(lam - 4*np.sqrt(max(lam, 1)))),
                        int(lam + 4*np.sqrt(max(lam, 1))) + 1)
    poisson_probs = poisson.pmf(k_range, lam)

    ax.bar(k_range, poisson_probs, width=0.8, color=COLORS['primary'],
           alpha=0.6, label=f'Poisson(λ={lam})')

    x_cont = np.linspace(k_range[0], k_range[-1], 300)
    normal_pdf = norm.pdf(x_cont, loc=lam, scale=np.sqrt(lam))
    ax.plot(x_cont, normal_pdf, color=COLORS['highlight'], linewidth=2.5,
            label=f'N(μ={lam}, σ²={lam})')

    ax.set_title(f'λ = {lam}', fontsize=12)
    ax.set_xlabel('k')
    ax.set_ylabel('P(k)')
    ax.legend(fontsize=8)

plt.suptitle('Poisson → Normal: The Gaussian Approximation Improves with λ',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("λ = 1:   Very skewed, clearly NOT Gaussian")
print("λ = 5:   Still asymmetric, Gaussian is a rough fit")
print("λ = 10:  Getting close — the 'rule of thumb' minimum for Gaussian approximation")
print("λ = 30:  Good match — most practical sensor scenarios")
print("λ = 100: Excellent — typical industrial camera pixel")
print("λ = 500: Indistinguishable — this is why we treat bright pixels as Gaussian")
