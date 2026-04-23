# Chapter 5 — Modeling Distributions

## The Question

> "Can we describe this entire distribution with just 2 numbers?"

So far we have described distributions empirically — we plot the actual data.
But empirical descriptions have a problem: they are specific to this sample.
If we collect new data, the histogram shifts slightly.

A **parametric model** says: "this data comes from a known family of distributions,
characterized by a small number of parameters." If the model fits well, we can:

- Describe the distribution in 2–3 numbers instead of thousands
- Generate synthetic data
- Compute probabilities analytically
- Compare datasets on the same scale

---

## The Exponential Distribution

**When it appears:** waiting times between events, inter-arrival times, survival times.

$$f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0$$

**Parameters:** one — the rate $\lambda$ (events per unit time), or equivalently the mean $\mu = 1/\lambda$.

**Key property:** memoryless. The probability of waiting another $t$ minutes is the same
regardless of how long you've already waited. Like a coin flip — the past is irrelevant.

**In NSFG:** the time between pregnancies follows an approximately exponential distribution.

### Detecting Exponential Shape
If $X \sim \text{Exponential}(\lambda)$, then on a log-y axis, the CDF becomes a straight line:
$$\ln(1 - F(x)) = -\lambda x$$

If the complementary CDF is linear on a log scale, the data is exponential.

---

## The Normal Distribution

The most famous distribution in statistics. Also called the **Gaussian distribution**.

$$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Parameters:** mean $\mu$ and standard deviation $\sigma$.

**Characterized by:** bell shape, symmetric around the mean, 68-95-99.7 rule:
- 68% of data within 1$\sigma$ of the mean
- 95% within 2$\sigma$
- 99.7% within 3$\sigma$

**In NSFG:** birth weight is approximately normal. Adult heights are normal.
Many measurement errors are normal (by the Central Limit Theorem — Chapter 14).

### Normal Probability Plot

To check if data is normally distributed, plot the data against what you would expect
if it were perfectly normal:

1. Sort your data: $x_1 \leq x_2 \leq \ldots \leq x_n$
2. Compute the expected normal quantiles: $q_i = \Phi^{-1}(i/n)$ where $\Phi^{-1}$ is the inverse normal CDF
3. Plot $x_i$ vs $q_i$

If the data is normal, the plot is a straight line. Curves indicate skew or heavy tails.

---

## The Lognormal Distribution

If $\ln(X)$ is normally distributed, then $X$ is **lognormal**.

$$f(x; \mu, \sigma) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$$

**When it appears:** anything that is the product of many independent factors.
Income, city populations, file sizes, biological growth rates.

**In NSFG:** birth weight is roughly normal, but adult body weight is more lognormal
(long right tail — a few very heavy people, no symmetric left tail).

---

## The Pareto Distribution

Named after economist Vilfredo Pareto. Describes the "80/20 rule" phenomenon.

$$f(x; x_m, \alpha) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m$$

**Parameters:** minimum value $x_m$ and shape $\alpha$.

**When it appears:** wealth, city sizes, earthquake magnitudes, word frequencies.

**Key property:** heavy right tail — extreme values are much more common than a
normal distribution predicts. The mean may be infinite (when $\alpha \leq 1$).

**Detecting Pareto shape:** on a log-log plot, the CDF becomes linear.

---

## Why Model?

Three reasons:

1. **Compression:** a fitted normal replaces thousands of data points with $(\mu, \sigma)$
2. **Interpolation:** estimate probabilities between observed values
3. **Communication:** "the data is approximately normal with $\mu = 7.4$, $\sigma = 1.2$" is instantly understood

But models can be wrong. Always check the fit visually. A model that fits poorly
is worse than no model — it gives false confidence.

---

## Exercises

1. Fit a normal distribution to birth weight. What are $\mu$ and $\sigma$?
2. Make a normal probability plot for birth weight. Does the fit look good?
3. Plot the complementary CDF of inter-pregnancy intervals on a log scale. Is it exponential?
4. Fit a lognormal distribution to income (if available) or birth weight. Which fits better?
5. What fraction of babies weigh more than $\mu + 2\sigma$ under the normal model? Verify against the data.

---

## Glossary

**parametric model** — a family of distributions described by a fixed number of parameters

**exponential distribution** — models waiting times; memoryless; characterized by rate $\lambda$

**normal distribution** — bell-shaped, symmetric; characterized by mean $\mu$ and std $\sigma$

**lognormal distribution** — distribution whose logarithm is normal; right-skewed

**Pareto distribution** — heavy-tailed distribution; models 80/20 phenomena

**normal probability plot** — scatter plot of data quantiles vs normal quantiles; linear if normal

**goodness of fit** — how well a model matches the observed data

**68-95-99.7 rule** — for a normal distribution, 68/95/99.7% of data falls within 1/2/3 std devs
