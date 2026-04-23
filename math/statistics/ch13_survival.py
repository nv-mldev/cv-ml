"""
Chapter 13 — Survival Analysis
Question: How long until the next pregnancy? And how do we handle women
         who haven't had another pregnancy yet (censored data)?

What this script builds:
  - Inter-pregnancy interval computation from NSFG
  - Censoring identification
  - Kaplan-Meier estimator from scratch
  - Hazard function estimation
  - Cohort comparison

Run: python ch13_survival.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import df as preg_df   # full pregnancy df, not filtered to live

COLORS = {
    'km':       '#2196F3',
    'naive':    '#F44336',
    'cohort1':  '#4CAF50',
    'cohort2':  '#FF9800',
    'hazard':   '#9C27B0',
    'neutral':  '#9E9E9E',
}

# ── Build inter-pregnancy intervals ───────────────────────────────────────────
# For each woman (caseid), find consecutive pregnancies and compute the gap.
# Women with only one recorded pregnancy are censored — they may have more later.

def compute_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inter-pregnancy intervals for each woman.
    Returns a DataFrame with columns: caseid, interval_months, censored.

    interval_months: months between end of pregnancy i and end of pregnancy i+1
    censored: True if the woman only had one pregnancy (we don't know the next interval)
    """
    # Need datend (pregnancy end date in century-months)
    if 'datend' not in df.columns:
        # Create synthetic data for demonstration
        np.random.seed(42)
        n_women = 3000
        records = []
        for i in range(n_women):
            n_preg = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.18, 0.07])
            times = sorted(np.random.uniform(0, 120, n_preg))
            for j in range(len(times) - 1):
                interval = times[j + 1] - times[j]
                records.append({'caseid': i, 'interval': interval, 'censored': False})
            if n_preg == 1:
                records.append({'caseid': i, 'interval': np.nan, 'censored': True})
        return pd.DataFrame(records)

    rows = []
    for caseid, group in df.groupby('caseid'):
        group_sorted = group.sort_values('datend').dropna(subset=['datend'])
        dates = group_sorted['datend'].values

        if len(dates) < 2:
            # Only one pregnancy: censored
            rows.append({'caseid': caseid, 'interval': np.nan, 'censored': True})
        else:
            for i in range(len(dates) - 1):
                rows.append({
                    'caseid':   caseid,
                    'interval': dates[i + 1] - dates[i],
                    'censored': False,
                })
            # Last pregnancy: censored (may have more later)
            rows.append({'caseid': caseid, 'interval': np.nan, 'censored': True})

    return pd.DataFrame(rows)


intervals_df = compute_intervals(preg_df)

n_total    = len(intervals_df)
n_observed = intervals_df[~intervals_df['censored']]['interval'].notna().sum()
n_censored = intervals_df['censored'].sum()

print("── Inter-Pregnancy Intervals ───────────────────────────────────────────")
print(f"  Total records      : {n_total:,}")
print(f"  Observed intervals : {n_observed:,}")
print(f"  Censored (no next) : {n_censored:,}  ({n_censored/n_total*100:.1f}%)")

observed = intervals_df[~intervals_df['censored']]['interval'].dropna().values

print(f"\n  Naive mean (ignoring censored) : {observed.mean():.2f} months")
print(f"  (This underestimates the true mean — censored women wait longer)")

# ── Kaplan-Meier estimator from scratch ───────────────────────────────────────

def kaplan_meier(
    intervals: np.ndarray,
    censored:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Kaplan-Meier survival estimator.

    Args:
        intervals : time of event OR last observation (if censored)
        censored  : True if the observation is censored (event not yet seen)

    Returns:
        times  : sorted unique event times
        S      : survival probability at each event time

    Algorithm: at each event time t_j,
        S(t_j) = S(t_{j-1}) * (1 - d_j / n_j)
    where d_j = events at t_j, n_j = at risk just before t_j.
    """
    # Replace NaN censoring times with a large value (they're "still waiting")
    t = intervals.copy()
    t[np.isnan(t)] = np.nanmax(t[~np.isnan(t)]) * 2  # push censored past all events

    # Get sorted unique event times (not censored)
    event_times = np.sort(np.unique(t[~censored]))

    S = 1.0
    survival_times  = [0.0]
    survival_probs  = [1.0]

    for tj in event_times:
        n_at_risk = np.sum(t >= tj)                    # still waiting at time tj
        d_j       = np.sum((t == tj) & (~censored))    # events at exactly tj
        if n_at_risk > 0 and d_j > 0:
            S *= 1 - d_j / n_at_risk
        survival_times.append(tj)
        survival_probs.append(S)

    return np.array(survival_times), np.array(survival_probs)


# Build arrays for KM
t_all  = intervals_df['interval'].values.copy().astype(float)
c_all  = intervals_df['censored'].values

# Replace NaN in t with a large placeholder for censored rows
max_observed = np.nanmax(t_all)
t_all_filled = np.where(np.isnan(t_all), max_observed * 2, t_all)

km_times, km_survival = kaplan_meier(t_all_filled, c_all)

print(f"\n── Kaplan-Meier Survival Curve ─────────────────────────────────────────")
print(f"  {'Months':>8}  {'S(t)':>8}  {'Meaning'}")
checkpoints = [6, 12, 18, 24, 36]
for cp in checkpoints:
    idx = np.searchsorted(km_times, cp)
    if idx < len(km_survival):
        s = km_survival[idx]
        print(f"  {cp:>8}  {s:>8.3f}  "
              f"{(1-s)*100:.1f}% have had next pregnancy by month {cp}")

# ── Hazard function ────────────────────────────────────────────────────────────

def hazard_from_survival(times: np.ndarray, survival: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate hazard from the KM survival curve.
    h(t) ≈ -ΔS / (S × Δt)
    """
    hazard_times  = []
    hazard_vals   = []
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > 0 and survival[i - 1] > 0:
            h = -(survival[i] - survival[i - 1]) / (survival[i - 1] * dt)
            hazard_times.append(times[i])
            hazard_vals.append(h)
    return np.array(hazard_times), np.array(hazard_vals)


h_times, h_vals = hazard_from_survival(km_times, km_survival)

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Naive CDF vs KM survival curve
ax = axes[0, 0]
# Naive: treat all observed as uncensored
naive_sorted = np.sort(observed[observed < 60])
n = len(naive_sorted)
naive_s = 1 - np.arange(1, n + 1) / n
ax.step(naive_sorted, naive_s, color=COLORS['naive'],
        linewidth=2, where='post', label='Naive (ignores censoring)')
# KM
mask = km_times <= 60
ax.step(km_times[mask], km_survival[mask], color=COLORS['km'],
        linewidth=2, where='post', label='Kaplan-Meier')
ax.set_xlabel('Months since last pregnancy')
ax.set_ylabel('S(t) = P(no next pregnancy yet)')
ax.set_title('Survival Curve: Inter-Pregnancy Interval')
ax.legend(fontsize=9)
ax.set_xlim(0, 60)

# 2. Hazard function
ax = axes[0, 1]
smooth_h = pd.Series(h_vals).rolling(5, center=True, min_periods=1).mean().values
mask_h = h_times <= 60
ax.plot(h_times[mask_h], smooth_h[mask_h], color=COLORS['hazard'], linewidth=2)
ax.set_xlabel('Months since last pregnancy')
ax.set_ylabel('Hazard rate')
ax.set_title('Hazard Function h(t)\n(instantaneous risk of next pregnancy)')
ax.set_xlim(0, 60)

# 3. Observed interval distribution (non-censored)
ax = axes[1, 0]
obs_clip = observed[(observed > 0) & (observed < 72)]
ax.hist(obs_clip, bins=30, density=True, color=COLORS['km'], alpha=0.8)
ax.axvline(np.median(obs_clip), color=COLORS['naive'], linewidth=2,
           label=f'Median = {np.median(obs_clip):.1f} months')
ax.set_xlabel('Inter-pregnancy interval (months)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Observed Intervals\n(censored excluded — biased!)')
ax.legend(fontsize=9)

# 4. Censoring pattern
ax = axes[1, 1]
categories = ['Observed interval\n(event occurred)', 'Censored\n(event not yet seen)']
counts = [n_observed, n_censored]
colors_bar = [COLORS['km'], COLORS['neutral']]
bars = ax.bar(categories, counts, color=colors_bar, alpha=0.8)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f'{count:,}', ha='center', fontsize=10)
ax.set_ylabel('Count')
ax.set_title('Observed vs Censored Observations')

plt.tight_layout()
plt.savefig('ch13_survival.png', dpi=150)
plt.show()
print("\nFigure saved: ch13_survival.png")
