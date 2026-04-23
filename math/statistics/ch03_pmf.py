"""
Chapter 3 — Probability Mass Functions
Question: What is the exact empirical distribution, without arbitrary bin choices?

What this script builds:
  - PMF class from scratch (just a normalized dict)
  - PMF plots for first vs other babies
  - The class size paradox — same data, two different answers
  - Size-biased distribution

Run: python ch03_pmf.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import first, other, live

COLORS = {
    'first':     '#2196F3',
    'other':     '#4CAF50',
    'highlight': '#F44336',
    'neutral':   '#9E9E9E',
    'paradox':   '#9C27B0',
}

# ── PMF: the core data structure ──────────────────────────────────────────────

class Pmf:
    """
    Probability Mass Function — maps values to probabilities.
    Built from a sequence of observations; probabilities sum to 1.
    """

    def __init__(self, values):
        counts: dict = {}
        for v in values:
            if not _is_nan(v):
                counts[v] = counts.get(v, 0) + 1
        total = sum(counts.values())
        self.d: dict = {v: c / total for v, c in counts.items()}

    def prob(self, value, default: float = 0.0) -> float:
        """Probability of a specific value."""
        return self.d.get(value, default)

    def values(self):
        return sorted(self.d.keys())

    def probs(self):
        return [self.d[v] for v in self.values()]

    def mean(self) -> float:
        """E[X] = sum of (value × probability)."""
        return sum(v * p for v, p in self.d.items())

    def __repr__(self):
        return f"Pmf({len(self.d)} values, mean={self.mean():.3f})"


def _is_nan(v) -> bool:
    try:
        return np.isnan(v)
    except (TypeError, ValueError):
        return False


# ── Build PMFs for pregnancy length ───────────────────────────────────────────

first_pmf = Pmf(first['prglngth'].dropna().values)
other_pmf = Pmf(other['prglngth'].dropna().values)

print("── PMF: Pregnancy Length ───────────────────────────────────────────────")
print(f"  {first_pmf}")
print(f"  Most probable value (first) : {max(first_pmf.d, key=first_pmf.d.get)} weeks")
print(f"  Most probable value (other) : {max(other_pmf.d, key=other_pmf.d.get)} weeks")

# Show difference at each week
print(f"\n  {'Week':>4}   {'First':>7}   {'Other':>7}   {'Diff':>8}")
for week in range(35, 45):
    f = first_pmf.prob(week)
    o = other_pmf.prob(week)
    bar = '█' * int(abs(f - o) * 500)
    print(f"  {week:>4}   {f:>7.4f}   {o:>7.4f}   {f-o:>+8.4f}  {bar}")

# ── The Class Size Paradox ─────────────────────────────────────────────────────
# Two ways to ask "what is the average class size?"
# Both use the same data — but the answers are very different.

print("\n── The Class Size Paradox ──────────────────────────────────────────────")

# Simulate a college with departments of various sizes
np.random.seed(42)
# 8 small departments (10 students) + 2 large (100 students)
departments = [10] * 8 + [100] * 2
students_per_dept = departments

dept_mean = np.mean(departments)
print(f"  Departments view: mean class size = {dept_mean:.1f} students")

# A random student is more likely to be in a large class
# Probability of landing in a class of size x ∝ x
all_students = []
for size in departments:
    all_students.extend([size] * size)

student_mean = np.mean(all_students)
print(f"  Students' view  : mean class size = {student_mean:.1f} students")
print(f"  Ratio           : {student_mean / dept_mean:.1f}x larger from students' perspective!")

# Same paradox in NSFG: ask the mother her total pregnancies
# Mothers with more pregnancies appear more often in the dataset
birthord_pmf_raw  = Pmf(live['birthord'].dropna().values)

# Size-biased version: weight each value by the value itself
def size_biased(pmf: Pmf) -> Pmf:
    """Simulate sampling one member from a group proportional to group size."""
    biased_counts = {}
    for v, p in pmf.d.items():
        biased_counts[v] = p * v
    total = sum(biased_counts.values())
    result = Pmf.__new__(Pmf)
    result.d = {v: c / total for v, c in biased_counts.items()}
    return result

birthord_pmf_biased = size_biased(birthord_pmf_raw)

print(f"\n  NSFG birth order paradox:")
print(f"  Mean birth order (raw PMF)    : {birthord_pmf_raw.mean():.3f}")
print(f"  Mean birth order (size-biased): {birthord_pmf_biased.mean():.3f}")
print(f"  A random child in the dataset 'experiences' a larger family than average.")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. PMF comparison: first vs other
ax = axes[0]
weeks = range(35, 45)
first_p = [first_pmf.prob(w) for w in weeks]
other_p = [other_pmf.prob(w) for w in weeks]
x = np.array(list(weeks))
ax.bar(x - 0.2, first_p, width=0.4, color=COLORS['first'], alpha=0.8, label='First')
ax.bar(x + 0.2, other_p, width=0.4, color=COLORS['other'], alpha=0.8, label='Other')
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('Probability')
ax.set_title('PMF: Pregnancy Length')
ax.legend()

# 2. PMF difference
ax = axes[1]
diffs = [first_pmf.prob(w) - other_pmf.prob(w) for w in weeks]
colors_diff = [COLORS['first'] if d > 0 else COLORS['other'] for d in diffs]
ax.bar(list(weeks), diffs, color=colors_diff, alpha=0.8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('P(first) - P(other)')
ax.set_title('PMF Difference (First − Other)')

# 3. Class size paradox
ax = axes[2]
values = sorted(birthord_pmf_raw.d.keys())
raw_probs   = [birthord_pmf_raw.prob(v)    for v in values]
biased_probs = [birthord_pmf_biased.prob(v) for v in values]
x = np.array(values)
ax.bar(x - 0.2, raw_probs,    width=0.4, color=COLORS['neutral'],  alpha=0.8, label='Raw PMF')
ax.bar(x + 0.2, biased_probs, width=0.4, color=COLORS['paradox'],  alpha=0.8, label='Size-biased')
ax.set_xlabel('Birth order')
ax.set_ylabel('Probability')
ax.set_title('Class Size Paradox: Birth Order')
ax.legend()

plt.tight_layout()
plt.savefig('ch03_pmf.png', dpi=150)
plt.show()
print("\nFigure saved: ch03_pmf.png")
