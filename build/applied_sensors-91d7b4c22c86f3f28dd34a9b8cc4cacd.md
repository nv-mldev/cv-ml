# Probability Applied — A Worked Sensor Noise Model
### From One Random Event to a Gaussian Noise Floor

> *Part 6 laid out the generic measurement chain. This page works it through end-to-end for one concrete sensor — the camera — because every link in the chain corresponds to physically distinct hardware, making it the cleanest worked example.*
>
> *The same template applies to any sensor (vibration, audio, network telemetry); the cross-reference table at the end shows where each one plugs in. This is the applied capstone of the [probability series](README.md).*

---

## 1. Random Variables and Distributions — Quick Recap

A **random variable** is a function that maps an experiment's outcome to a number. Every sensor reading is a random variable — it takes a different value each time you measure the same underlying state, because the physics of measurement is inherently random.

A **probability distribution** completely describes a random variable: it assigns a probability to every possible value.

The three quantities that characterise a distribution:
- **Mean** $\mu = E[X]$ — the expected value
- **Variance** $\sigma^2 = E[(X-\mu)^2]$ — the spread
- **Standard deviation** $\sigma$ — same units as $X$

---

## 2. The Distribution Chain — Worked for the Camera Sensor

The camera is the cleanest physical instance of the chain because each link corresponds to a different piece of silicon. The same chain applies to any sensor — only the physical interpretations of $p$ and $\lambda$ change.

### Bernoulli — one photon, one photosite

Each photon either reaches the photosite or it doesn't. Probability $p$ of detection (= quantum efficiency, QE):

$$P(X=1) = p, \quad P(X=0) = 1-p$$

Mean = $p$, Variance = $p(1-p)$.

*Generic version:* one event, two outcomes, probability $p$ of "success."

### Binomial — $n$ photons, one photosite

Count the number of photons detected in $n$ independent trials:

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Mean = $np$, Variance = $np(1-p)$.

*Generic version:* count successes in $n$ independent identical trials.

### Poisson — photon flux model

When $n \to \infty$ and $p \to 0$ such that $np = \lambda$ (mean count) stays fixed:

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Mean = $\lambda$, Variance = $\lambda$ (mean = variance — the defining property of Poisson).

This is the exact model for photon arrivals over an exposure. The parameter $\lambda$ is the average photon count per exposure.

*Generic version:* rare events arriving at rate $\lambda$ per window — photons per exposure, shock pulses per second, packet drops per minute.

### Normal (Gaussian) — large count limit

When $\lambda \gg 1$ (bright scene, many photons), the Poisson distribution converges to a Gaussian:

$$\text{Poisson}(\lambda) \xrightarrow{\lambda \to \infty} \mathcal{N}(\mu=\lambda,\; \sigma^2=\lambda)$$

**This is why shot noise is Gaussian** in bright conditions: it is the large-$\lambda$ limit of Poisson photon counting.

### Central Limit Theorem — why sums are always Gaussian

The CLT says: the sum of any $n$ independent random variables with finite mean and variance converges to a Gaussian as $n \to \infty$, regardless of the original distributions.

In a sensor, multiple noise sources add together (shot noise + read noise + dark current + quantization). Even if each source is non-Gaussian, their sum is approximately Gaussian. The CLT is why we can model total sensor noise as Gaussian in almost all practical cases — and why the same conclusion holds for a vibration sensor, an audio ADC, or any other measurement front-end.

---

## 3. Shot Noise Is Irreducible

From the Poisson model: $\sigma_{shot} = \sqrt{\lambda}$.

This cannot be reduced by better hardware — it is a fundamental property of the discrete nature of light. To halve the shot noise you must quadruple the photon count (larger aperture, longer exposure, bigger photosite).

$$\text{SNR} = \frac{\lambda}{\sqrt{\lambda}} = \sqrt{\lambda}$$

Doubling the photon count improves SNR by $\sqrt{2} \approx 41\%$. This is the fundamental physical limit for any photon-counting measurement.

The same $\sigma = \sqrt{\lambda}$ rule holds for any Poisson process: bearing-fault counts, packet drops, Geiger counter clicks. In each case, the irreducible relative noise is $1/\sqrt{\lambda}$ — the only way to reduce it is to count more events.

---

## 4. Noise Budget

A real sensor has multiple noise sources. The total noise variance is their sum (assuming independence):

$$\sigma_{total}^2 = \sigma_{shot}^2 + \sigma_{read}^2 + \sigma_{dark}^2 + \sigma_{quant}^2$$

$$= \lambda + \sigma_r^2 + \sigma_d^2 + \frac{\Delta^2}{12}$$

where $\Delta$ is the quantization step size.

**Which term dominates depends on the signal regime:**

| Regime | Dominant noise | Characteristic |
|---|---|---|
| Bright ($\lambda \gg \sigma_r^2$) | Shot noise | $\sigma \propto \sqrt{\lambda}$ |
| Dim ($\lambda \ll \sigma_r^2$) | Read noise | $\sigma \approx \sigma_r$ (constant floor) |
| Saturated ($\lambda \geq C$) | Clipping | All information above $C$ is lost |

In bright conditions, collecting more signal always helps. In dim conditions, electronics dominate and more signal doesn't help much until you climb out of the electronics floor.

The same regime structure shows up in any measurement chain: an electronics-limited floor at low signal, a physics-limited (often $\sqrt{\lambda}$) regime at medium signal, and a saturation regime where the measurement is destroyed.

---

## 5. The Anscombe Transform

Poisson noise has variance equal to its mean — which makes it awkward statistically (the noise level depends on the signal level). The **Anscombe transform** converts Poisson noise to approximately unit-variance Gaussian:

$$A(X) = 2\sqrt{X + \frac{3}{8}}$$

After applying $A$, the data has approximately constant variance $\approx 1$ regardless of $\lambda$. This lets standard Gaussian-assumption algorithms work correctly on Poisson-distributed data — useful any time you have count data you want to feed into a tool that assumes constant-variance Gaussian noise.

> **Simulation:** the [probability series](README.md) — `part0` through `part6` — develops every step on this page from scratch using Monte Carlo simulation. Read those first if you want the build-up; this page is the destination.

---

## 6. The Same Chain — Where Other Sensors Plug In

| Stage | Camera | Vibration (accelerometer) | Network telemetry | Audio (microphone) |
|---|---|---|---|---|
| **Bernoulli event** | Photon detected (QE) | Shock pulse in a time slice | Packet dropped or not | Pressure crossing threshold |
| **Binomial count** | Photons detected per exposure | Pulses per window | Drops per minute | Threshold crossings per frame |
| **Poisson regime** | $\lambda$ = expected photons | $\lambda$ = expected pulses/hour | $\lambda$ = expected drops/min | (rarely the dominant model) |
| **Gaussian limit** | High illumination | Many pulses per window | High-traffic monitoring | Almost always — the noise floor is Gaussian to begin with |
| **Electronics floor** | Read noise | Charge-amp + ADC noise | (none) | Pre-amp + ADC noise |
| **Quantization** | ADC bit depth | ADC bit depth | (count, no quantization) | 16/24-bit ADC |

The chain is the same. Only the names change.

---

## Summary

| Concept | Key fact |
|---|---|
| Bernoulli | One trial, two outcomes |
| Binomial | $n$ trials, count of successes |
| Poisson | Limiting case $n\to\infty$, $p\to 0$, $np=\lambda$; mean = variance = $\lambda$ |
| Gaussian | Limiting case $\lambda\to\infty$; why most noise floors are approximately Normal |
| CLT | Sum of any independent RVs → Gaussian; why total sensor noise is Normal |
| SNR | $= \sqrt{\lambda}$ in any Poisson regime (photons, pulses, packets) |
| Noise budget | $\sigma^2_{total} = \sigma^2_{physics} + \sigma^2_{electronics} + \sigma^2_{quantization}$ |

---

**Next →** [Linear algebra applied — Images as vectors](../linear_algebra/applied_images.md): the normalisation operations in Chapter 6 (L2 norm, dot product, mean subtraction) have a precise geometric interpretation that the linear algebra track makes explicit.
