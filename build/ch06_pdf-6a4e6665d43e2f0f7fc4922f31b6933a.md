# Chapter 6 — Probability Density Functions

## From Discrete to Continuous

So far we've worked with discrete distributions (PMF) and empirical CDFs.
But many real variables are conceptually continuous — birth weight, height, income.

For a continuous variable, the probability of any exact value is zero:
$$P(X = 7.4321\ldots) = 0$$

Instead, we ask: **what is the probability of falling in a small interval?**

$$P(a \leq X \leq b) = \int_a^b f(x)\, dx$$

Here $f(x)$ is the **probability density function (PDF)**. It is not a probability —
it is a density. A value of $f(x) = 2.5$ means "the probability per unit length at $x$ is 2.5."

---

## From Histogram to PDF

A histogram is an approximation of the PDF — but it depends on bin width.

The **kernel density estimate (KDE)** is a smooth histogram that doesn't require
you to choose bins. Instead, it places a smooth "kernel" (usually Gaussian) at
each data point and sums them:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)$$

where $h$ is the **bandwidth** (controls smoothness) and $K$ is the kernel function.

The bandwidth is the only free parameter. Too small → noisy, too large → oversmoothed.
Most libraries choose $h$ automatically using Silverman's rule of thumb:

$$h = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5}$$

---

## The Distribution Framework

Now we have four ways to represent a distribution:

| Representation | What it answers | When to use |
|---|---|---|
| **Histogram** | rough shape | first look at data |
| **PMF** | $P(X = x)$ for discrete $X$ | categorical or integer data |
| **CDF** | $P(X \leq x)$ | percentiles, comparing groups |
| **PDF/KDE** | shape of continuous distribution | smooth visualization, modeling |

They all contain the same information — just organized differently.

**Relations:**
- PMF → CDF: cumulative sum
- CDF → PDF: derivative
- PDF → CDF: integral

---

## Moments

A **moment** is a summary statistic computed from powers of the data.

**First moment (mean):**
$$\mu = E[X] = \int x\, f(x)\, dx$$

**Second moment (variance):**
$$\sigma^2 = E[(X - \mu)^2] = \int (x - \mu)^2 f(x)\, dx$$

**Third standardized moment (skewness):**
$$\text{skewness} = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$

- Positive skewness: long right tail (income, city sizes)
- Negative skewness: long left tail
- Zero skewness: symmetric (normal distribution)

**Fourth standardized moment (kurtosis):**
$$\text{kurtosis} = E\!\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3$$

- Positive kurtosis (leptokurtic): heavier tails than normal
- Negative kurtosis (platykurtic): lighter tails
- Zero kurtosis: normal distribution

---

## Skewness in NSFG

Birth weight has slight **negative skewness** — there are more very low-weight
babies (premature, small for gestational age) than very high-weight ones.

Pregnancy length has **negative skewness** too — you can be very premature
(27 weeks) but not very post-term (almost never past 44 weeks).

Mother's age has slight **positive skewness** — most mothers are in their 20s,
but the tail extends into the 40s.

---

## Exercises

1. Plot a KDE for birth weight. Try three different bandwidths. How does the shape change?
2. Compute skewness for birth weight, pregnancy length, and mother's age.
3. Which NSFG variable has the most positive skewness? The most negative?
4. Plot the CDF and its derivative (approximate PDF) for birth weight on the same figure.
5. Implement a Gaussian KDE from scratch using the kernel formula above.

---

## Glossary

**PDF (Probability Density Function)** — $f(x)$ such that $P(a \leq X \leq b) = \int_a^b f(x)\,dx$

**density** — value of $f(x)$; not a probability, but probability per unit length

**KDE (Kernel Density Estimate)** — smooth empirical PDF estimated from data

**bandwidth** — smoothing parameter in KDE; controls trade-off between noise and smoothness

**Silverman's rule** — data-driven default bandwidth: $h = 1.06 \hat\sigma n^{-1/5}$

**moment** — $E[X^k]$; the $k$-th moment summarizes a distribution's shape

**skewness** — third standardized moment; measures asymmetry

**kurtosis** — fourth standardized moment; measures tail heaviness relative to normal
