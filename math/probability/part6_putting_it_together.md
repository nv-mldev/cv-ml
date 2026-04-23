# Part 6: Putting It All Together — The Complete Image Noise Model in CV Pipelines

## The Full Noise Budget

Now we can understand every noise source in a camera sensor and why the total noise ends up Gaussian:

| Noise Source | Physical Origin | Distribution | Parameters |
|-------------|----------------|--------------|------------|
| **Shot noise** | Photon arrival is quantum-random | Poisson($\lambda$) | $\lambda$ = expected photons |
| **Dark current** | Thermal electrons (no light needed) | Poisson($\lambda_d$) | $\lambda_d$ = dark electrons (∝ temp × time) |
| **Read noise** | Amplifier + ADC electronics | Gaussian($0, \sigma_r^2$) | $\sigma_r$ ≈ 1–5 electrons (modern sensors) |
| **Quantization noise** | ADC rounding to integers | Uniform($-\Delta/2, \Delta/2$) | $\Delta$ = step size = $\frac{\text{full well}}{2^{\text{bits}}}$ |
| **JPEG compression artifacts** | Quantization error in DCT coefficients | Approximately Uniform | $\pm$ quantization step / 2 |
| **Transmission noise** | Bit errors, packet loss | Sparse / Impulse | Salt-and-pepper pattern |

**Total signal in electrons:**

$$S = \underbrace{\text{Poisson}(\lambda)}_{\text{photons}} + \underbrace{\text{Poisson}(\lambda_d)}_{\text{dark current}} + \underbrace{\mathcal{N}(0, \sigma_r^2)}_{\text{read noise}} + \underbrace{\text{Uniform}(-\Delta/2, \Delta/2)}_{\text{quantization}}$$

By the CLT, this sum of independent random variables approaches Gaussian with:
- Mean: $\mu = \lambda + \lambda_d$
- Variance: $\sigma^2 = \lambda + \lambda_d + \sigma_r^2 + \Delta^2/12$

---

## The Complete Signal Chain (Simulation Function)

The full sensor simulation builds a pixel value from first principles:

1. **Photon arrival**: draw from `Poisson(λ)` — the quantum noise floor
2. **Quantum efficiency**: thin each photon count through `Binomial(n_photons, QE)` — not every photon produces a signal
3. **Dark current**: add `Poisson(dark_current)` electrons — accumulate even with no light
4. **Read noise**: add `Normal(0, read_noise_σ)` — electronic noise from the readout circuit
5. **Clipping**: clip to [0, full_well_capacity] — the sensor cannot count more than its capacity
6. **ADC quantization**: scale to $[0, 2^{\text{bit\_depth}} - 1]$ and round to integers

This model is what NLM, BM3D, and Wiener filtering exploit — they are designed assuming exactly this noise structure.

**Key observations from the signal chain:**
1. Shot noise (Poisson) dominates at this photon count
2. QE reduces the signal but preserves the Poisson shape
3. Dark current adds a small amount of extra Poisson noise
4. Read noise barely changes the shape (small σ relative to shot noise)
5. The total is well-approximated by Gaussian (by the CLT)
6. ADC quantisation creates discrete steps but doesn't change the envelope

---

## The Three Regimes of Sensor Noise

Depending on the signal level, different noise sources dominate:

| Regime | Signal Level | Dominant Noise | SNR Behavior |
|--------|-------------|----------------|--------------|
| **Read-noise limited** | Dark pixels | Electronics ($\sigma_r$) | SNR ∝ signal (linear) |
| **Shot-noise limited** | Mid-tones | Physics ($\sqrt{\lambda}$) | SNR ∝ $\sqrt{\text{signal}}$ |
| **Saturation** | Bright pixels | Well is full | SNR collapses |

**Why this matters:** knowing which regime you are in tells you what denoising strategy to use:
- In the **read-noise regime**, averaging frames helps (reduce $\sigma_r$ by $1/\sqrt{n}$)
- In the **shot-noise regime**, you need signal-dependent denoising (Anscombe transform + Gaussian denoiser)
- In the **saturation regime**, information is irreversibly lost

---

## Simulate an Image Capture: The Noise Model Behind CV Denoising Algorithms

Understanding this noise model is what makes algorithms like **Non-Local Means (NLM)**, **BM3D**, and **Wiener filtering** work — they exploit the statistical structure of the noise: Poisson shot noise at low signal, transitioning to Gaussian at higher signal levels.

**What to look for in the gradient simulation:**
- Left edge: grainy (read-noise limited, low SNR)
- Middle: smooth Poisson shot noise
- Right edge: saturates to white

This is exactly the image a denoising algorithm like NLM or BM3D receives as input.

---

## Key Takeaways

1. **Bernoulli → Binomial → Poisson → Normal** is a chain of increasing abstraction. Each distribution is a limiting case of the one before it, and each simplification makes the math easier while preserving the essential statistics.

2. **Photon counting is Poisson — and so are defect counts, keypoint counts, and any rare-event counting in images.** The Poisson distribution's key property — variance equals mean ($\sigma^2 = \lambda$) — means that noise is signal-dependent: bright pixels are cleaner (relatively) than dark pixels.

3. **The CLT is the reason Gaussian assumptions work in CV.** SIFT descriptor components, HOG histograms, Gaussian blur outputs, background subtraction models, and denoising algorithms all rely on the fact that sums of many independent contributions converge to Normal.

4. **Every CV algorithm that assumes Gaussian noise is implicitly relying on this entire chain.** When you call `cv2.GaussianBlur` or fit a `GaussianMixture`, you are exploiting the Bernoulli → Binomial → Poisson → Normal chain.

5. **Three noise regimes** exist in every sensor:
   - **Read-noise limited** (dark pixels): electronics dominate, SNR ∝ signal
   - **Shot-noise limited** (mid-tones): physics dominates, SNR ∝ √signal
   - **Saturated** (bright pixels): information is lost, SNR collapses
