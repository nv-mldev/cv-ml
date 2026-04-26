# Chapter 7 — Probability for Sensors
### Why Sensor Noise Is Gaussian, and How to Budget It

> *Chapter 2 asserted that photon arrivals are a Poisson process and that pixel noise is approximately Gaussian. This chapter derives both claims from first principles, building the distribution chain: Bernoulli → Binomial → Poisson → Normal → CLT.*

---

## 7.1 Random Variables and Distributions

A **random variable** is a function that maps an experiment's outcome to a number. Every pixel value is a random variable — it takes a different value each time you photograph the same scene, because photon arrivals are random.

A **probability distribution** completely describes a random variable: it assigns a probability to every possible value.

The three quantities that characterise a distribution:
- **Mean** $\mu = E[X]$ — the expected value
- **Variance** $\sigma^2 = E[(X-\mu)^2]$ — the spread
- **Standard deviation** $\sigma$ — same units as $X$

---

## 7.2 The Distribution Chain

The path from "one photon either arrives or it doesn't" to "the pixel noise is Gaussian":

### Bernoulli — one photon, one photosite

Each photon either reaches the photosite or it doesn't. Probability $p$ of detection (= quantum efficiency):

$$P(X=1) = p, \quad P(X=0) = 1-p$$

Mean = $p$, Variance = $p(1-p)$.

### Binomial — $n$ photons, one photosite

Count the number of photons detected in $n$ independent trials:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Mean = $np$, Variance = $np(1-p)$.

### Poisson — photon flux model

When $n \to \infty$ and $p \to 0$ such that $np = \lambda$ (mean count) stays fixed:

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Mean = $\lambda$, Variance = $\lambda$ (mean = variance — the defining property of Poisson).

This is the exact model for photon arrivals. The parameter $\lambda$ = average photon count per exposure.

### Normal (Gaussian) — large count limit

When $\lambda \gg 1$ (bright scene, many photons), the Poisson distribution converges to a Gaussian:

$$\text{Poisson}(\lambda) \xrightarrow{\lambda \to \infty} \mathcal{N}(\mu=\lambda,\; \sigma^2=\lambda)$$

**This is why shot noise is Gaussian** in bright conditions: it is the large-$\lambda$ limit of Poisson photon counting.

### Central Limit Theorem — why sums are always Gaussian

The CLT says: the sum of any $n$ independent, identically distributed random variables with finite mean and variance converges to a Gaussian as $n \to \infty$, regardless of the original distribution.

In a sensor, multiple noise sources add together (shot noise + read noise + dark current). Even if each source is non-Gaussian, their sum is approximately Gaussian. The CLT is why we can model total sensor noise as Gaussian in almost all practical cases.

---

## 7.3 Shot Noise Is Irreducible

From the Poisson model: $\sigma_{shot} = \sqrt{\lambda}$.

This cannot be reduced by better hardware — it is a fundamental property of the discrete nature of light. To halve the shot noise you must quadruple the photon count (larger aperture, longer exposure, bigger photosite).

$$\text{SNR} = \frac{\lambda}{\sqrt{\lambda}} = \sqrt{\lambda}$$

Doubling the photon count improves SNR by $\sqrt{2} \approx 41\%$. This is the fundamental limit.

---

## 7.4 Noise Budget

A real sensor has multiple noise sources. The total noise variance is their sum (assuming independence):

$$\sigma_{total}^2 = \sigma_{shot}^2 + \sigma_{read}^2 + \sigma_{dark}^2 + \sigma_{quant}^2$$

$$= \lambda + \sigma_r^2 + \sigma_d^2 + \frac{\Delta^2}{12}$$

where $\Delta$ is the quantization step size.

**Which term dominates depends on the brightness regime:**

| Regime | Dominant noise | Characteristic |
|--------|---------------|----------------|
| Bright ($\lambda \gg \sigma_r^2$) | Shot noise | $\sigma \propto \sqrt{\lambda}$ |
| Dim ($\lambda \ll \sigma_r^2$) | Read noise | $\sigma \approx \sigma_r$ (constant floor) |
| Saturated ($\lambda \geq C$) | Clipping | All information above $C$ is lost |

In bright conditions, shooting more light always helps. In dim conditions, read noise dominates and more photons don't help much until you are out of the read-noise floor.

---

## 7.5 The Anscombe Transform

Poisson noise has variance equal to its mean — which makes it hard to work with statistically (the noise level depends on the signal level). The **Anscombe transform** converts Poisson noise to approximately unit-variance Gaussian:

$$A(X) = 2\sqrt{X + \frac{3}{8}}$$

After applying $A$, the data has approximately constant variance $\approx 1$ regardless of $\lambda$. This enables standard Gaussian-assumption algorithms to work correctly on Poisson-distributed data.

> **Simulation:** `~/projects/cv-ml/math/probability/`
> — All 6 parts: Bernoulli → Binomial → Poisson → Normal → CLT → putting it together.

---

## Summary

| Concept | Key fact |
|---------|----------|
| Bernoulli | One photon: detected or not |
| Binomial | $n$ photons: count of detections |
| Poisson | Limiting case $n\to\infty$; mean = variance = $\lambda$ |
| Gaussian | Limiting case $\lambda\to\infty$; why shot noise is approximately Normal |
| CLT | Sum of any independent RVs → Gaussian; why total sensor noise is Normal |
| SNR | $= \sqrt{\lambda}$ in shot-noise regime |
| Noise budget | $\sigma^2_{total} = \sigma^2_{shot} + \sigma^2_{read} + \sigma^2_{dark} + \sigma^2_{quant}$ |

---

**Next →** [Chapter 8 — Linear Algebra for Images](../ch08_linear_algebra/README.md): the normalisation operations in Chapter 6 (L2 norm, dot product, mean subtraction) have a precise geometric interpretation that the linear algebra framework makes explicit.
