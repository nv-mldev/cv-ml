# Chapter 14 — Analytic Methods

## The Question

> "We've been simulating everything. When can we use formulas instead?"

All previous chapters built understanding through simulation — we shuffled,
resampled, and generated. But many textbooks give you formulas directly.

When are those formulas valid? And when does simulation win?

---

## Normal Distributions — The Special Case

Many analytic results only hold when data is normally distributed.

If $X \sim N(\mu, \sigma^2)$ and $Y \sim N(\nu, \tau^2)$ independently, then:

$$X + Y \sim N(\mu + \nu, \sigma^2 + \tau^2)$$

Sums of normals are normal. This is special — it doesn't hold for most distributions.

---

## Sampling Distributions — The Key Results

If we draw samples of size $n$ from a population with mean $\mu$ and std $\sigma$:

**Mean:**
$$\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{(exactly if population is normal)}$$

**Difference in means:**
$$\bar{X}_1 - \bar{X}_2 \sim N\!\left(\mu_1 - \mu_2, \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}\right)$$

This is how the classical two-sample t-test is derived — it assumes both populations
are normal and uses the above formula for the sampling distribution.

---

## The Central Limit Theorem (CLT)

The most important theorem in statistics.

**Statement:** For any population with finite mean $\mu$ and variance $\sigma^2$,
as $n \to \infty$:

$$\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1)$$

The sample mean is approximately normally distributed, **regardless of the shape
of the population distribution**, as long as $n$ is large enough.

**"Large enough"** depends on skewness:
- Symmetric population → $n \approx 30$ is usually fine
- Moderately skewed → $n \approx 100$
- Heavily skewed → $n \approx 500+$

**Why does this matter?** It is the reason normal-based formulas work in so many
situations even when the data is not normal — we're taking means of large samples,
and means are approximately normal.

---

## Testing the CLT

With NSFG birth weight (slightly left-skewed):
- Population is NOT normal
- Draw samples of size $n$, compute sample mean
- As $n$ increases, the distribution of sample means should approach normal

We can verify this by simulation — plot the sampling distribution of the mean
for $n = 5, 20, 100, 500$. Watch it converge to normal.

---

## Applying the CLT — The t-test

For the first-baby pregnancy length question, instead of a permutation test (Chapter 9),
we could use a **two-sample t-test**:

$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

Under $H_0$, this follows a t-distribution with approximately:

$$\nu \approx \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1-1) + (s_2^2/n_2)^2/(n_2-1)}$$

degrees of freedom (Welch's approximation, no assumption of equal variance).

**Does it give the same answer as the permutation test?** Almost always yes,
when $n$ is large. This validates both approaches.

---

## Correlation Test

For testing whether Pearson's $r$ is significantly different from zero:

Under $H_0: \rho = 0$:

$$t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} \sim t_{n-2}$$

We can compute this analytically instead of running a permutation test — and
for large $n$, the answers match.

---

## Chi-Squared Test — Analytic Form

In Chapter 9 we introduced chi-squared for categorical data. The analytic version:

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i} \sim \chi^2_{k-1}$$

under $H_0$, where $k$ is the number of categories.

---

## When Simulation Beats Formulas

Use simulation (permutation test, bootstrap) when:
1. The data is heavily skewed or has outliers (CLT hasn't kicked in yet)
2. The test statistic is not the mean (e.g., median, max, ratio)
3. You have small samples ($n < 30$)
4. The analytic formula requires assumptions you can't verify

Use analytic methods when:
1. $n$ is large and data is not extremely skewed
2. You need speed (simulation is 1000× slower)
3. You want closed-form confidence intervals
4. You need to communicate to an audience that expects p-values and t-statistics

---

## Exercises

1. Simulate the CLT: draw samples of size $n = 5, 20, 100, 500$ from birth weight. Plot sampling distributions of the mean.
2. Run a two-sample t-test for pregnancy length (first vs other). Compare to the permutation test p-value.
3. Compute the analytic 95% confidence interval for mean birth weight. Compare to the bootstrap CI.
4. Test the correlation between age and birth weight analytically. Same result as the permutation test?
5. For which NSFG variable does the sampling distribution of the mean converge slowest to normal? Why?

---

## Glossary

**Central Limit Theorem (CLT)** — the sample mean is approximately normal for large $n$, regardless of population shape

**t-distribution** — the sampling distribution of the mean when $\sigma$ is unknown; approaches normal as $n \to \infty$

**t-test** — hypothesis test based on the t-statistic; assumes (approximately) normal sampling distribution

**Welch's approximation** — formula for degrees of freedom in two-sample t-test without assuming equal variances

**chi-squared distribution** — distribution of $\sum (O-E)^2/E$ under $H_0$ for categorical data

**analytic method** — a formula-based result derived from probability theory (vs simulation-based)

**normal approximation** — using a normal distribution to approximate the sampling distribution when the CLT applies

**convergence in distribution** — a sequence of distributions approaching a limit distribution
