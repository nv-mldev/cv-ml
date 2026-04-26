# Chapter 3 — Probability Mass Functions

## The Problem with Histograms

In Chapter 2, we normalized histograms to compare groups of different sizes.
But histograms have a hidden problem: **the shape changes with bin width**.

Try these bin widths for pregnancy length:
- 1 week → clean picture, matches clinical intuition
- 0.5 weeks → noisy, looks random
- 3 weeks → too coarse, merges distinct peaks

There is no objectively correct bin width. This is uncomfortable.

The **Probability Mass Function (PMF)** solves this for discrete data.

---

## What is a PMF?

A PMF maps each possible value to its probability:

$$P(X = x) = \frac{\text{count}(x)}{n}$$

For pregnancy length (measured in whole weeks), there are about 18 distinct values
(27 through 44). The PMF assigns a probability to each one.

No bins. No arbitrary choices. The PMF is the exact empirical distribution.

**Key property:** probabilities sum to 1.

$$\sum_x P(X = x) = 1$$

---

## Building a PMF from Scratch

```python
def make_pmf(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    total = sum(counts.values())
    return {v: count / total for v, count in counts.items()}
```

That's it. A PMF is just a normalized frequency count.

---

## Plotting PMFs

Unlike a histogram (bars touching, continuous), a PMF for discrete data is
plotted as **separated bars** or **stems** — because the gaps between integer
values are real: no pregnancy is 38.7 weeks long.

---

## PMF Reveals What Histogram Hides

When you plot first vs other babies as PMFs, you see something a histogram
smooths over: the distributions have subtly different shapes near 39 weeks.

Other babies are slightly more likely to be born at exactly 39 weeks.
First babies have a longer right tail.

The mean difference (13 hours) is real, but it's driven by the tail, not a
uniform shift of the whole distribution.

---

## The Class Size Paradox

This is one of the most elegant examples in statistics — a case where
**the same data gives different answers depending on who you ask**.

**Setup:** A college has departments of varying sizes:

| Department | Size | # Departments |
|---|---|---|
| Small (10 students) | 10 | 8 |
| Medium (100 students) | 100 | 2 |

From the **department's perspective:**
- Average class size = (8 × 10 + 2 × 100) / 10 = 28 students

From the **student's perspective:**
- Most students are in large classes
- A student picked at random is more likely to be in a 100-person class
- Average class size experienced by a student = (80 × 10 + 200 × 100) / 280 ≈ 79 students

Same college. Same data. Completely different averages. Why?

Because the probability of being in a large class is proportional to the
class size itself. The PMF of class sizes is biased by the very thing we're measuring.

**Formally:** if $P(X = x)$ is the PMF of class sizes, the size-biased distribution is:

$$P(X^* = x) = \frac{x \cdot P(X = x)}{E[X]}$$

This is called a **size-biased distribution** or the **inspection paradox**.

**In NSFG:** the same paradox applies to family size. If you ask mothers how
many children they have, you oversample mothers with many children
(because large families contribute more respondents).

---

## DataFrame Indexing

When computing PMFs from a DataFrame, we often want to select rows by value:

```python
# Fraction of pregnancies that are first babies (among live births)
pmf = make_pmf(live['birthord'].dropna().values)
prob_first = pmf.get(1, 0)
```

Or compare subsets:

```python
for birthord in [1, 2, 3, 4]:
    group = live[live['birthord'] == birthord]
    mean_len = group['prglngth'].mean()
    print(f"Birth order {birthord}: mean length = {mean_len:.3f} weeks")
```

---

## Exercises

1. Build a PMF of `prglngth` for all live births. What is the most common length?
2. Build PMFs for first vs other babies separately. At which week do they differ most?
3. Implement the class size paradox using NSFG family size data.
4. Build a PMF of birth order (`birthord`). What fraction of live births are first babies?
5. What is the mean of the size-biased distribution of `birthord`? Compare to the raw mean.

---

## Glossary

**PMF (Probability Mass Function)** — maps each value of a discrete variable to its probability

**discrete variable** — a variable that takes countable values (like whole weeks)

**size-biased distribution** — a distribution where sampling probability is proportional to value size

**inspection paradox** — the phenomenon where the distribution experienced by a random member differs from the population distribution

**empirical distribution** — the distribution computed directly from observed data (as opposed to a theoretical model)
