# Part 5: The Central Limit Theorem — Why Everything Becomes Gaussian

## An Applied Scenario — Smoothing the Vibration Stream

Back to the running motor. The raw accelerometer stream is jagged — every sample is a noisy snapshot. To get a stable reading you do the obvious thing: take a **moving average** over the last $n$ samples.

Try it with $n = 10$. The output is calmer than the raw stream but still bumpy. Try $n = 100$. Calmer still. Try $n = 1000$ and the smoothed signal is almost glassy.

Two things happen as $n$ grows:

1. The **spread** of the smoothed signal shrinks. Doubling $n$ shrinks the spread by $\sqrt{2}$.
2. The **shape** of the smoothed signal's distribution converges to a clean bell curve — even if the raw samples have a weird, skewed, or bimodal distribution.

The first observation is intuitive (averaging cancels noise). The second is much deeper, and it's the most useful theorem in all of probability.

---

## Intuition

The Poisson-to-Normal convergence in Part 4 turned out to be a special case of something far more general: the **Central Limit Theorem** (CLT).

The CLT says: **if you sum or average many independent random variables, the result is approximately Normal — regardless of what distribution the individual variables came from.**

That last part is what makes it powerful. The original variables don't have to be Gaussian. They don't even have to be the *same* distribution. As long as you're adding up enough independent contributions of comparable size, the answer is a bell curve.

This single fact explains an enormous amount:

- **Moving-average filters** smooth any signal toward a Gaussian-distributed output
- **Sensor noise floors** are Gaussian because each reading is a sum of dozens of micro-disturbances
- **Calibration errors** are Gaussian because each measurement aggregates many small sources
- **Mini-batch gradients in deep learning** are approximately Gaussian around the true gradient — the optimization theory of SGD leans hard on this

---

## The Theorem

Let $X_1, X_2, \ldots, X_n$ be independent random variables, each with:
- Mean $\mu$
- Variance $\sigma^2$

Then as $n \to \infty$, the **sample mean** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ converges to:

$$\bar{X} \sim \mathcal{N}\left(\mu, \, \frac{\sigma^2}{n}\right)$$

Or equivalently, the **sum** $S_n = \sum_{i=1}^n X_i$ converges to:

$$S_n \sim \mathcal{N}\left(n\mu, \, n\sigma^2\right)$$

**The individual $X_i$ can be from ANY distribution** — uniform, exponential, Poisson, Bernoulli, even something bizarre and bimodal. As long as you sum enough of them, the result is Gaussian.

---

## Back to the Smoothed Vibration

For the moving-average filter on the raw accelerometer stream:

- $\sigma^2$ — variance of one raw sample
- $n$ — window length
- Smoothed sample variance: $\sigma^2 / n$
- Smoothed sample standard deviation: $\sigma / \sqrt{n}$

That's the **$1/\sqrt{n}$ noise reduction law**, and it falls out of the CLT for free. Want half the noise? Quadruple the window. Want a tenth of the noise? Multiply the window by 100.

It's also why a smoothed signal looks Gaussian-distributed even when the raw stream is wildly non-Gaussian (impulsive transients, clipped values, mixed regimes). Averaging always pulls you toward the bell.

---

## CLT Explains the Poisson → Normal Convergence

The Poisson($\lambda$) random variable can be thought of as the **sum of $\lambda$ independent Poisson(1) variables**:

$$\text{Poisson}(\lambda) = \underbrace{\text{Poisson}(1) + \text{Poisson}(1) + \cdots + \text{Poisson}(1)}_{\lambda \text{ terms}}$$

By the CLT, this sum converges to $\mathcal{N}(\lambda, \lambda)$ as $\lambda \to \infty$. The Poisson-to-Normal convergence from Part 4 is just one instance of the same theorem.

This is not just mathematical elegance — it's why a single Gaussian noise model works for any sensor at moderate-to-high signal levels, regardless of whether the underlying physics is photon counting, packet arrivals, or something else.

---

## A Sensor Reading is Already a Sum

A single reading from almost any sensor is itself a sum of many independent contributions:

$$\text{reading} = \underbrace{\text{primary signal}}_{\text{the thing you want}} + \underbrace{\sum_{i} \eta_i}_{\text{thermal / electronic noise}} + \underbrace{\sum_{j} d_j}_{\text{interference / drift}} + \underbrace{q}_{\text{quantization}}$$

Each $\eta_i$ and $d_j$ is itself a sum of many micro-disturbances. By the CLT, the total error is approximately Gaussian even though the individual sources have completely different distributions.

This is the deeper reason every classical signal-processing tool — Wiener filtering, Kalman filtering, Gaussian denoising, least-squares estimation — assumes Gaussian noise. The CLT guarantees it for any reading that aggregates enough small independent error sources.

---

## CLT Convergence Rate

The convergence to Normal is fast for some distributions and slow for others. The rate depends on the **skewness** of the original distribution:

- **Uniform** (symmetric): converges fastest — already symmetric, just needs smoothing
- **Bernoulli** (discrete): moderate convergence
- **Exponential** (skewed): converges slower
- **Bimodal** (two peaks): slowest — two peaks need the most averaging to merge

The convergence rate is approximately $O(1/\sqrt{n})$. In practical terms, this tells you how long a moving-average window needs to be before the smoothed output is genuinely Gaussian, or how many frames to average before a background-subtraction model based on a Gaussian assumption is valid.

**What to look for in the KS convergence plot:** all curves decrease, but at different rates. The Kolmogorov–Smirnov (KS) statistic measures the maximum difference between the empirical CDF and the standard Normal CDF. A KS distance below ~0.02 means the distribution is effectively Gaussian for any downstream tool that assumes it.
