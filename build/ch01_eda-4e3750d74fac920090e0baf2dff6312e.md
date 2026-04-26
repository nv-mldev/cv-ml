# Chapter 1 — Exploratory Data Analysis

## The Reversal

In probability, you **knew the distribution** and asked: *what data would it produce?*

In statistics, you **have the data** and ask: *what distribution produced it?*

This reversal is the entire subject. You already understand random variables,
Bernoulli trials, and what a distribution means. Now we point that knowledge
at real data and ask hard questions.

---

## The Question

> "Everyone knows first babies are born late."

You've heard this. Your parents believe it. It gets repeated at baby showers.
But is it actually true? And how would you know?

This is the central question of Chapter 1 — and of statistics itself.
The goal is not to trust intuition. The goal is to **replace anecdote with evidence**.

---

## A Statistical Approach

The trap most people fall into: they remember the cases that confirm their belief
and forget the ones that don't. This is called **confirmation bias**, and it's
built into human cognition.

Statistics gives us a way out:

1. Find data collected independently of our belief
2. Define the question precisely (what does "late" mean?)
3. Compute something measurable
4. Decide if the difference is real or just noise

Step 4 is what most of this course is about. But first, we need data.

---

## The Dataset — NSFG

The **National Survey of Family Growth** is a US government survey conducted by
the CDC. The 2002 cycle interviewed 7,643 women aged 15–44 about their pregnancies,
births, and health. It is designed to be representative of the US population.

Why this dataset?

- Large enough to detect small effects
- Collected by professionals with no agenda about first babies
- Contains exactly the variables we need: pregnancy length, birth order, birth weight

### Key Variables

| Variable | Description |
|---|---|
| `caseid` | Unique ID for each respondent |
| `prglngth` | Pregnancy length in weeks |
| `outcome` | 1 = live birth, 2 = induced abortion, 3 = stillbirth, ... |
| `birthord` | Birth order (1 = first baby, 2 = second, ...) |
| `agepreg` | Mother's age at pregnancy end (in centiyears — see below) |
| `totalwgt_lb` | Birth weight in pounds |
| `finalwgt` | Survey weight (how many women this respondent represents) |

---

## Loading the Data

The NSFG data comes in a fixed-width format (`.dat.gz`) described by a dictionary
file (`.dct`). We parse it from scratch — no helper libraries.

See `ch01_eda.py` for the full loading code.

---

## Transformation

Raw NSFG data needs cleaning before it is usable:

### Age in centiyears
`agepreg` is stored as age × 100 (e.g., 2550 means 25.5 years).
```python
df['agepreg'] = df['agepreg'] / 100.0
```

### Filter live births only
`outcome == 1` means live birth. Everything else (abortion, stillbirth, etc.)
is excluded when studying birth weight and pregnancy length.

```python
live_births = df[df['outcome'] == 1]
```

### Split by birth order
```python
first_babies = live_births[live_births['birthord'] == 1]
other_babies  = live_births[live_births['birthord'] > 1]
```

---

## Validation

Before trusting any analysis, check the data against known published statistics.

From the NSFG codebook:
- Total pregnancies: 13,593
- Live births: 9,148
- Mean pregnancy length (live births): ~38.6 weeks

If our numbers don't match, something went wrong in loading or filtering.
This is not optional — **always validate before analyzing**.

---

## First Look — Does the Effect Exist?

```
Mean pregnancy length:
  First babies : 38.601 weeks
  Other babies : 38.523 weeks
  Difference   :  0.078 weeks  ≈  13 hours
```

There is a difference. First babies are born slightly later.

But 13 hours — is that "late"? Is it real? Could it be random noise?

These are the questions the next 13 chapters are built to answer.

---

## Interpretation

The key lesson of this chapter: **a difference in means is just the beginning.**

We need to ask:
- How large is the difference relative to the variability? (Chapter 2)
- Could this happen by chance? (Chapter 9)
- Is it practically meaningful? (Chapter 2 — effect size)

The answer to "are first babies born late?" turns out to be:
**yes, slightly, but the effect is much smaller than people believe.**

---

## Exercises

1. Load the NSFG data and print the shape of the DataFrame.
2. How many live births are in the dataset?
3. What is the mean birth weight for first babies vs. others?
4. What fraction of pregnancies result in live births?
5. Plot a histogram of `agepreg` for the full dataset. What shape do you see?

---

## Glossary

**anecdotal evidence** — a single story or personal experience used to support a general claim

**confirmation bias** — the tendency to notice and remember cases that confirm your beliefs

**DataFrame** — a 2D table in pandas with named columns and an index

**outcome variable** — in NSFG, the coded result of a pregnancy (live birth, abortion, etc.)

**survey weight** — a number that accounts for the fact that some groups were oversampled

**centiyear** — one hundredth of a year; used in NSFG to store ages as integers

**validation** — checking computed results against independently known values to catch errors
