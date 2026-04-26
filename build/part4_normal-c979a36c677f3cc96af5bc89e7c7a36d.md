# Part 4: The Normal (Gaussian) Distribution — The Bell Curve

## An Applied Scenario — The Vibration Sensor With No Motor Running

Switch the motor off. The accelerometer keeps streaming, and even though there's nothing to measure, the readings aren't exactly zero — they wander around it.

Plot a few seconds of these "silent" samples and you get a tight cloud centered on 0 g with some spread:

- A handful of samples sit near ±0.001 g
- Most cluster tightly around 0
- Almost none stray past ±0.005 g

The cloud is symmetric and bell-shaped. There's no obvious physical "rare event" being counted here — instead, every reading is a sum of dozens of tiny disturbances: thermal noise in the amplifier, electromagnetic pickup, ADC quantization, mechanical vibration from the building. None of them dominate; they just add up.

When that's the situation — many small independent contributions summed together — the resulting distribution has a name, and a shape you've already glimpsed in the Binomial and Poisson plots from earlier parts.

---

## Intuition

Look back at the Binomial plots from Part 2. As $n$ grew, the distribution started looking like a smooth bell curve. The Poisson does the same as $\lambda$ grows. Sums of many small noise sources do the same thing — that's the silent-sensor cloud above.

The **Normal distribution** (or Gaussian) is the continuous bell-shaped curve that all of these converge to. It is the most important distribution in statistics, signal processing, and machine learning — not because nature is "secretly Gaussian," but because **anything that's a sum of many small independent things ends up Gaussian**. (We'll prove this in Part 5 — the Central Limit Theorem.)

The Normal shows up wherever sums show up:

- **Sensor noise floors** — many independent micro-disturbances summed
- **Filter outputs** — convolution averages many input samples
- **Calibration errors** — repeated measurements of the same quantity
- **Aggregated metrics** — mean of a batch, sum over a window
- **Pixel intensities in uniform regions** — independent noise sources per pixel

---

## The Math

$$f(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}} \, e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

Two parameters:
- $\mu$ = mean (center of the bell)
- $\sigma^2$ = variance (width of the bell)
- $\sigma$ = standard deviation

Unlike Binomial and Poisson (which are **discrete** — only integer counts), the Normal is **continuous** — it assigns probability density to any real number.

**Why this matters in practice:** once you've decided a process is Normal, the entire distribution is captured by two numbers. Estimating $\mu$ and $\sigma$ from data is straightforward, and almost every classical signal-processing or denoising algorithm — Wiener filtering, Kalman filtering, Gaussian blur, BM3D denoising — assumes its inputs or noise are Gaussian.

---

## Back to the Silent Sensor

Fit a Gaussian to the silent-sensor samples and you'll get something like:

$$\mu \approx 0, \quad \sigma \approx 0.002 \quad \text{(both in g)}$$

That single number $\sigma$ is your **noise floor** — the irreducible spread of the sensor when there's no signal. Any future "signal" you measure has to be large enough relative to $\sigma$ to be distinguishable from this floor. The signal-to-noise ratio (SNR) is built directly on top of this.

Anything more than a few $\sigma$ from the mean is suspicious: under a Normal model, $|x - \mu| > 3\sigma$ happens about $0.27\%$ of the time. If you start seeing $5\sigma$ excursions on the silent sensor, the noise model itself has changed — temperature drift, a loose cable, a new EMI source.

---

## Poisson → Normal as $\lambda$ Grows

For large $\lambda$, the Poisson distribution is well-approximated by a Normal with the same mean and variance:

$$\text{Poisson}(\lambda) \approx \mathcal{N}(\mu = \lambda, \, \sigma^2 = \lambda) \quad \text{for } \lambda \gg 1$$

This is the link back to the bearing-fault and photon-counting examples from Part 3. At low counts (a handful of clicks per hour, a few photons per pixel) the discreteness matters and you need the actual Poisson PMF. At high counts — bright illumination, busy networks, high event rates — the Gaussian approximation takes over and life gets analytically easier.

**Practical threshold:** for $\lambda > 30$, the Normal approximation is tight enough for most engineering purposes. This is why noise-modelling algorithms (Wiener, Kalman, NLM, BM3D) get away with assuming Gaussian noise — at moderate-to-high signal levels, Poisson has already become Gaussian.

---

## The Complete Chain

```
                    n → ∞, p → 0                  λ → ∞
Bernoulli(p) → Binomial(n, p) ────────→ Poisson(λ) ────────→ Normal(λ, λ)
   │                │                       │                      │
   │                │                       │                      │
 1 trial,       n trials,              expected count          Gaussian
 yes/no?        k successes              λ = np              approximation
                                    (variance = mean)      (analytically nice)
```

Each arrow is a mathematical limit that simplifies the model while preserving the essential statistics.

This chain shows up across signal processing: binary threshold decisions (Bernoulli) → counts in a window (Binomial) → rare-event rates like faults, packet drops, photon arrivals (Poisson) → Gaussian approximation enabling tractable downstream algorithms (Normal). Whatever the domain, you're walking the same ladder.

The next part shows *why* the Normal sits at the end of this chain — and why it's the end of the chain for almost every other distribution too. That's the Central Limit Theorem.
