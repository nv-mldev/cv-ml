# Part 4: The Normal (Gaussian) Distribution — The Bell Curve

## Intuition

Look back at the Binomial plots from Part 2. As $n$ increased, the distribution started looking like a smooth bell curve. The Poisson does the same thing as $\lambda$ increases.

The **Normal distribution** is the continuous bell-shaped curve that both the Binomial and Poisson approach under the right conditions. It is the most important distribution in statistics — and in computer vision.

The Normal arises throughout CV because:
- **Filter outputs:** convolving with many pixels → CLT → Gaussian output distribution
- **Pixel intensities in uniform regions:** many independent noise sources add up → Gaussian
- **Feature descriptor components:** HOG, SIFT gradients averaged over many pixels → CLT → approximately Normal
- **Calibration measurement errors:** repeated measurements of the same point → Gaussian

## The Math

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

Two parameters:
- $\mu$ = mean (center of the bell)
- $\sigma^2$ = variance (width of the bell)
- $\sigma$ = standard deviation

Unlike Binomial and Poisson (which are **discrete** — only integer counts), the Normal is **continuous** — it assigns probability to any real number.

**Why this matters for CV:** Gaussian blur, SIFT/HOG descriptor components, and background subtraction models all assume pixel values or filter outputs are Normal. The two parameters $\mu$ and $\sigma$ are all you need to describe them.

---

## Poisson → Normal as $\lambda$ Grows

For large $\lambda$, the Poisson distribution is well-approximated by a Normal with the same mean and variance:

$$\text{Poisson}(\lambda) \approx \mathcal{N}(\mu = \lambda, \, \sigma^2 = \lambda) \quad \text{for } \lambda \gg 1$$

This is crucial for sensor engineering and CV algorithms: at high photon counts (or high signal levels), we can treat shot noise as **Gaussian** — which is analytically much easier to work with.

**This is why denoising algorithms (NLM, BM3D, Wiener filtering) use Gaussian noise models** — they are valid at moderate-to-high signal levels, where Poisson noise has already converged to Gaussian.

**Practical threshold:** for $\lambda > 30$, the Normal approximation is tight enough for most CV applications — this is why denoising algorithms (NLM, BM3D, Wiener) use Gaussian noise models at moderate-to-high signal levels.

---

## The Complete Chain

```
                    n → ∞, p → 0                  λ → ∞
Bernoulli(p) → Binomial(n, p) ────────→ Poisson(λ) ────────→ Normal(λ, λ)
   │                │                       │                      │
   │                │                       │                      │
 1 photon      n photons,             Expected count          Gaussian
 detected?     k detected               λ = np              approximation
                                    (variance = mean)      (analytically nice)
```

Each arrow is a mathematical limit that simplifies the model while preserving the essential statistics.

This chain appears throughout CV: binary pixel decisions (Bernoulli) → counting pixels above threshold (Binomial) → modeling rare events like defects or keypoints (Poisson) → Gaussian approximation enabling tractable algorithms for SIFT, HOG, background subtraction, and denoising.
