"""
Chapter 1 — Exploratory Data Analysis
Question: Are first babies born later than others?

What this script does:
  - Parses NSFG fixed-width data from scratch (no helper libraries)
  - Cleans and transforms key variables
  - Validates against published NSFG statistics
  - Computes the first-baby effect: mean pregnancy length difference

Run: python ch01_eda.py
Requires: python data/download_nsfg.py  (run once first)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import gzip
import os

# ── Color palette (consistent across all chapters) ────────────────────────────
COLORS = {
    'first':     '#2196F3',   # blue  — first babies
    'other':     '#4CAF50',   # green — other babies
    'highlight': '#F44336',   # red   — differences, annotations
    'neutral':   '#9E9E9E',   # grey  — background elements
}

# ── Step 1: Parse the .dct dictionary file ────────────────────────────────────
# The .dct file describes column positions and types for the fixed-width .dat file
# Format example:  _column(str5)  caseid  1-12   "Case ID number"

def parse_dct(dct_path: str) -> list[dict]:
    """Read a Stata .dct file and return column specs as a list of dicts."""
    columns = []
    pattern = re.compile(r"_column\((\w+)\)\s+(\w+)\s+(\d+)-(\d+)")
    with open(dct_path) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                dtype_str, name, start, end = match.groups()
                columns.append({
                    'name':  name,
                    'start': int(start) - 1,   # convert to 0-indexed
                    'end':   int(end),
                    'dtype': dtype_str,
                })
    return columns


def load_fixed_width(dat_path: str, columns: list[dict]) -> pd.DataFrame:
    """Parse a gzipped fixed-width file using column specs from parse_dct."""
    records = []
    opener = gzip.open if dat_path.endswith('.gz') else open
    with opener(dat_path, 'rt') as f:
        for line in f:
            record = {}
            for col in columns:
                raw = line[col['start']:col['end']].strip()
                if raw == '':
                    record[col['name']] = np.nan
                elif col['dtype'].startswith('str'):
                    record[col['name']] = raw
                else:
                    try:
                        record[col['name']] = float(raw)
                    except ValueError:
                        record[col['name']] = np.nan
            records.append(record)
    return pd.DataFrame(records)


# ── Step 2: Load NSFG pregnancy data ──────────────────────────────────────────

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
dct_path = os.path.join(data_dir, '2002FemPreg.dct')
dat_path = os.path.join(data_dir, '2002FemPreg.dat.gz')

if not os.path.exists(dat_path):
    print("ERROR: Data files not found.")
    print("Run: python data/download_nsfg.py")
    exit(1)

print("Loading NSFG pregnancy data...")
columns = parse_dct(dct_path)
df = load_fixed_width(dat_path, columns)
print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Step 3: Transformation ─────────────────────────────────────────────────────
# agepreg is stored as age × 100 (centiyears) — convert to years
df['agepreg'] = df['agepreg'] / 100.0

# birthwgt is stored as two separate columns: pounds + ounces
# Combine them into a single float: totalwgt_lb
if 'birthwgt_lb' in df.columns and 'birthwgt_oz' in df.columns:
    df['totalwgt_lb'] = df['birthwgt_lb'] + df['birthwgt_oz'] / 16.0

print("\nKey variable ranges after transformation:")
print(f"  agepreg     : {df['agepreg'].min():.1f} – {df['agepreg'].max():.1f} years")
print(f"  prglngth    : {df['prglngth'].min():.0f} – {df['prglngth'].max():.0f} weeks")

# ── Step 4: Filter to live births only ────────────────────────────────────────
# outcome == 1 means live birth
live = df[df['outcome'] == 1].copy()
print(f"\nLive births: {len(live)} out of {len(df)} total pregnancies")
print(f"  ({len(live)/len(df)*100:.1f}% of all pregnancies)")

# ── Step 5: Validation ────────────────────────────────────────────────────────
# Check against published NSFG statistics to confirm correct parsing

print("\n── Validation ──────────────────────────────────────────────────────")
print(f"  Total pregnancies  : {len(df):,}   (expected ~13,593)")
print(f"  Live births        : {len(live):,}   (expected ~9,148)")
print(f"  Mean prglngth      : {live['prglngth'].mean():.3f} weeks  (expected ~38.6)")

# ── Step 6: Split by birth order ──────────────────────────────────────────────
first = live[live['birthord'] == 1]
other = live[live['birthord'] > 1]

print(f"\n── Groups ──────────────────────────────────────────────────────────")
print(f"  First babies : {len(first):,}")
print(f"  Other babies : {len(other):,}")

# ── Step 7: The first-baby effect ─────────────────────────────────────────────
mean_first = first['prglngth'].mean()
mean_other = other['prglngth'].mean()
diff_weeks = mean_first - mean_other
diff_hours = diff_weeks * 7 * 24

print(f"\n── The First-Baby Effect ────────────────────────────────────────────")
print(f"  Mean pregnancy length (first babies) : {mean_first:.3f} weeks")
print(f"  Mean pregnancy length (other babies) : {mean_other:.3f} weeks")
print(f"  Difference                           : {diff_weeks:.3f} weeks")
print(f"                                       = {diff_hours:.1f} hours")
print(f"\n  Interpretation: First babies are born ~13 hours later on average.")
print(f"  But is this difference real or just noise? → Chapter 9 answers this.")

# ── Step 8: Visualisation ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pregnancy length distribution
ax = axes[0]
ax.hist(first['prglngth'], bins=range(27, 46), alpha=0.6,
        color=COLORS['first'], label='First babies', density=True)
ax.hist(other['prglngth'], bins=range(27, 46), alpha=0.6,
        color=COLORS['other'], label='Other babies', density=True)
ax.axvline(mean_first, color=COLORS['first'], linestyle='--', linewidth=1.5)
ax.axvline(mean_other, color=COLORS['other'], linestyle='--', linewidth=1.5)
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('Proportion')
ax.set_title('Pregnancy Length: First vs Other Babies')
ax.legend()

# Mother's age at pregnancy
ax = axes[1]
ax.hist(live['agepreg'].dropna(), bins=30, color=COLORS['neutral'],
        edgecolor='white', density=True)
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Proportion')
ax.set_title("Mother's Age at Pregnancy End")

plt.tight_layout()
plt.savefig('ch01_eda.png', dpi=150)
plt.show()
print("\nFigure saved: ch01_eda.png")
