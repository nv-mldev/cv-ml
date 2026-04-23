# Chapter 4 — Cumulative Distribution Functions

## The Limits of PMFs

PMFs work well for discrete data with few unique values (like pregnancy length in weeks).
But for continuous data — birth weight in pounds — there can be hundreds of distinct
values, and plotting every one gives a noisy, unreadable picture.

We need a representation that:
1. Works for any data (discrete or continuous)
2. Makes it easy to compare distributions visually
3. Supports questions like "what percentile is 8 lbs?"

The **Cumulative Distribution Function (CDF)** does all three.

---

## What is a CDF?

The CDF answers one question: **what fraction of the data is at or below a given value?**

$$F(x) = P(X \leq x)$$

For empirical data, this is simply:

$$F(x) = \frac{\text{number of values} \leq x}{n}$$

The CDF is always:
- Non-decreasing (once a value is "counted", it stays counted)
- Between 0 and 1
- Starts at 0, ends at 1

---

## Building a CDF from Scratch

```python
def make_cdf(values):
    sorted_values = sorted(v for v in values if not np.isnan(v))
    n = len(sorted_values)
    cumulative_prob = [(v, (i + 1) / n) for i, v in enumerate(sorted_values)]
    return cumulative_prob
```

Every value in the dataset gets a rank. The CDF at value $x$ is that rank divided by $n$.

---

## Percentiles

A **percentile** is the inverse CDF: given a probability $p$, find the value $x$
such that $F(x) = p$.

- 50th percentile = **median** — half the data is below this value
- 25th percentile = **first quartile** (Q1)
- 75th percentile = **third quartile** (Q3)
- IQR = Q3 − Q1 = the middle 50% of the data

For birth weight: the median is around 7.4 lbs. The 25th percentile is ~6.6 lbs,
the 75th is ~8.1 lbs. So the typical newborn weighs between 6.6 and 8.1 lbs.

---

## Comparing CDFs

The real power of the CDF: **two CDFs on the same plot tell the whole story**.

For pregnancy length:
- If the first-baby CDF is consistently to the right of the other-baby CDF,
  first babies are uniformly longer
- If the CDFs cross, the distributions differ in shape, not just location

When you plot the two CDFs for pregnancy length, they are nearly identical.
The 13-hour difference is truly tiny — the CDFs almost overlap completely.

---

## Percentile-Based Statistics

The median is more **robust** than the mean: one outlier doesn't shift it much.

For skewed data (like income, or birth weight which has a long left tail for
premature babies), the median is usually a better summary than the mean.

| Statistic | Formula | Sensitive to outliers? |
|---|---|---|
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | Yes |
| Median | value at 50th percentile | No |
| IQR | Q3 − Q1 | No |
| Std dev | $\sqrt{\frac{1}{n}\sum(x_i - \bar{x})^2}$ | Yes |

---

## Random Numbers from a CDF

A useful trick: if $U$ is a uniform random number between 0 and 1,
then $F^{-1}(U)$ has the same distribution as the original data.

This lets us **simulate new data** that matches an observed distribution:

```python
# Generate 1000 synthetic birth weights matching NSFG
u = np.random.uniform(0, 1, size=1000)
synthetic = [cdf_inverse(birth_weight_cdf, ui) for ui in u]
```

This is the foundation of the bootstrap (Chapter 8).

---

## Comparing Percentile Ranks

If you know a baby's birth weight, you might want to know: what percentile is this?

```python
# What fraction of babies weigh less than 8.0 lbs?
rank = cdf_value(birth_weight_cdf, 8.0)
print(f"A baby weighing 8.0 lbs is at the {rank*100:.1f}th percentile")
```

This is exactly what doctors use for growth charts.

---

## Exercises

1. Build a CDF for birth weight (`totalwgt_lb`) for all live births.
2. What is the median birth weight? What is the IQR?
3. Plot CDFs for first vs other babies on the same axes. What do you notice?
4. Write a function `percentile_rank(cdf, value)` that returns the percentile of a given value.
5. Write a function `percentile(cdf, p)` that returns the value at a given percentile.
6. Use the inverse CDF to generate 500 synthetic birth weights. Plot them vs the original CDF.

---

## Glossary

**CDF (Cumulative Distribution Function)** — $F(x) = P(X \leq x)$; fraction of data at or below $x$

**percentile** — the value below which a given percentage of observations fall

**median** — the 50th percentile; robust to outliers

**quartile** — 25th, 50th, or 75th percentile

**IQR (Interquartile Range)** — Q3 − Q1; the spread of the middle 50% of the data

**inverse CDF** — given a probability $p$, returns the value $x$ such that $F(x) = p$

**robust statistic** — a statistic that is not strongly affected by outliers
