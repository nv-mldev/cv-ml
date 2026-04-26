# Part 5: The Central Limit Theorem — Why Everything Becomes Gaussian

## Intuition

The Poisson-to-Normal convergence is a special case of something much more profound: the **Central Limit Theorem** (CLT).

The CLT says: **if you average many independent random variables, the result is approximately Normal — regardless of what distribution the individual variables came from.**

This is arguably the most important theorem in all of statistics. It explains why Gaussian assumptions work in so many CV algorithms:

- **Gaussian blur:** averaging neighboring pixels → CLT → output distribution is Gaussian
- **SIFT/HOG descriptors:** gradient histograms accumulated over many pixels → CLT → descriptor components approximately Normal
- **Background subtraction:** pixel value sampled over many frames → CLT → background model is Gaussian (justifies Gaussian Mixture Models)
- **Sensor averaging:** stack $n$ exposures → variance reduces as $1/n$ → total is Gaussian (CLT guarantees it)

## The Theorem

Let $X_1, X_2, \ldots, X_n$ be independent random variables, each with:
- Mean $\mu$
- Variance $\sigma^2$

Then as $n \to \infty$, the **sample mean** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ converges to:

$$\bar{X} \sim \mathcal{N}\left(\mu, \, \frac{\sigma^2}{n}\right)$$

Or equivalently, the **sum** $S_n = \sum_{i=1}^n X_i$ converges to:

$$S_n \sim \mathcal{N}\left(n\mu, \, n\sigma^2\right)$$

**The individual $X_i$ can be from ANY distribution** — uniform, exponential, Poisson, even something bizarre. As long as you sum enough of them, the result is Gaussian.

---

## Why This Matters for CV Algorithms

A pixel value is the **sum of many independent contributions**:

$$\text{pixel value} = \underbrace{\sum_{i=1}^{\lambda} 1}_{\text{photon arrivals (Poisson)}} + \underbrace{\sum_{j=1}^{m} \eta_j}_{\text{read noise (thermal)}} + \underbrace{\sum_{k=1}^{t} d_k}_{\text{dark current}}$$

Each of these is itself a sum of many independent random events. By the CLT, the total noise is approximately Gaussian — even though the individual sources have different distributions.

**This is why `cv2.GaussianBlur`, `sklearn.mixture.GaussianMixture`, and virtually every CV noise model assume Gaussian distributions — the CLT guarantees it.** When you average enough independent things, you get a bell curve. Period.

---

## CLT Explains the Poisson → Normal Convergence

The Poisson($\lambda$) random variable can be thought of as the **sum of $\lambda$ independent Poisson(1) variables**:

$$\text{Poisson}(\lambda) = \underbrace{\text{Poisson}(1) + \text{Poisson}(1) + \cdots + \text{Poisson}(1)}_{\lambda \text{ terms}}$$

By the CLT, this sum converges to $\mathcal{N}(\lambda, \lambda)$ as $\lambda \to \infty$. So the Poisson-to-Normal convergence we saw in Part 4 is just a specific instance of the CLT.

This is not just mathematical elegance — it's why engineers can use simple Gaussian noise models for cameras operating at reasonable photon counts.

---

## CLT Convergence Rate

The convergence rate depends on the **skewness** of the original distribution:

- **Uniform** (symmetric): converges fastest — already symmetric, just needs smoothing
- **Bernoulli** (discrete): moderate convergence
- **Exponential** (skewed): converges slower
- **Bimodal** (two peaks): slowest — two peaks need the most averaging to merge

The convergence rate is approximately $O(1/\sqrt{n})$ — this tells you how many frames to average in background subtraction before the Gaussian assumption is valid.

**What to look for in the KS convergence plot:** all curves decrease, but at different rates. The Kolmogorov-Smirnov (KS) statistic measures the maximum difference between the empirical CDF and the standard Normal CDF. A KS distance below ~0.02 indicates the distribution is effectively Gaussian.
