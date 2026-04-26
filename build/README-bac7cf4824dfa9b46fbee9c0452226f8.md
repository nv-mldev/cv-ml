# Statistics — From Data to Insight

A complete 14-chapter statistics course following Allen Downey's *Think Stats* structure,
built from scratch using NumPy and Pandas — no black-box helper classes.

## Where This Fits in the Curriculum

This folder is **not** the starting point. See `math/README.md` for the full sequence.

```
probability/part0 → part1          ← start here (randomness, Bernoulli)
        ↓
statistics/ch01 → ch02             ← jump here (real data + histograms)
        ↓
probability/part2 → part6          ← return here (distributions, motivated now)
        ↓
statistics/ch03 → ch14             ← finish here (inference, modeling, testing)
```

**The pivot:** after `ch02` (histograms), students ask *"what math model fits this shape?"*
That question drives them back into probability to learn Normal, Poisson, CLT.
Then they return here with the tools to go deeper.

## Dataset

**NSFG — National Survey of Family Growth (2002)**
A US government survey of ~13,000 women covering pregnancies, births, and health.
We use it to ask increasingly sophisticated questions across all 14 chapters.

```bash
python data/download_nsfg.py   # run once to download raw data files
```

## Chapter Map

| Chapter | File | Core Question | Tool Built |
|---------|------|---------------|------------|
| 1 | `ch01_eda.py` | Are first babies born late? | Data loading, cleaning |
| 2 | `ch02_distributions.py` | What does the distribution look like? | Histogram |
| 3 | `ch03_pmf.py` | How do we compare unequal groups? | PMF from scratch |
| 4 | `ch04_cdf.py` | What percentile is your birth weight? | CDF from scratch |
| 5 | `ch05_modeling.py` | Can 2 numbers describe this data? | Normal, Exponential, Pareto |
| 6 | `ch06_pdf.py` | What is the true shape? | KDE, moments, skewness |
| 7 | `ch07_relationships.py` | Does age predict birth weight? | Correlation, Covariance |
| 8 | `ch08_estimation.py` | How good is your guess? | Bootstrap, sampling dist |
| 9 | `ch09_hypothesis_testing.py` | Is the first-baby effect real? | Permutation test |
| 10 | `ch10_least_squares.py` | What's the best line? | OLS from scratch |
| 11 | `ch11_regression.py` | What predicts preterm birth? | Multiple + logistic regression |
| 12 | `ch12_time_series.py` | Has birth weight changed over time? | Autocorrelation, moving avg |
| 13 | `ch13_survival.py` | How long until next pregnancy? | Kaplan-Meier |
| 14 | `ch14_analytic_methods.py` | When can we skip simulation? | CLT, normal approximation |

## Learning Flow

Every chapter follows the same structure:

```
Practical Question → Intuition → Math → Code (from scratch) → Simulation → Interpretation
```

## Running a Chapter

```bash
cd /home/nithin/projects/cv-ml/math/statistics
python ch01_eda.py
```

## Philosophy

- **No black boxes.** We build PMF, CDF, and hypothesis tests ourselves so you see every step.
- **Same dataset, deeper questions.** Each chapter asks a harder question about the same data.
- **Simulation before formulas.** We simulate first, derive the formula second.
- **Effect size matters as much as p-value.** Statistical significance ≠ practical significance.
