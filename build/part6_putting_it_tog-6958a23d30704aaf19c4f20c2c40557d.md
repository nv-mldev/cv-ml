# Part 6: Putting It All Together — The Complete Measurement Noise Model

The whole probability ladder we've built — Bernoulli → Binomial → Poisson → Normal, glued together by the CLT — exists to answer one question that every signal-processing engineer faces:

> *Given a measurement, how much of what I'm seeing is signal and how much is noise — and what shape does that noise take?*

This part assembles the answer. We'll work it out for the camera sensor (the cleanest end-to-end example, because every link in the chain corresponds to physically distinct hardware), then show that the same template applies to any sensor — vibration, audio, network telemetry, anything.

---

## The Measurement Chain — Generic Template

Almost every digital sensor performs the same five-stage transformation from physical world to integer reading:

```
physical event   →   transduction   →   accumulation   →   electronics   →   ADC
   (random)         (each event has        (sum over           (read,         (round to
                     some detection         exposure /         amplify)         bits)
                     probability)           window)
```

Each stage adds its own statistical character:

| Stage | What's random | Distribution |
|---|---|---|
| Physical event | Quantum / thermal / arrival randomness | Often **Poisson** (rare, independent events) |
| Transduction | Each event detected with some probability | **Binomial** thinning of the count |
| Accumulation | Sum over a window | The Poisson stays Poisson; sums of small noises become **Normal** by CLT |
| Electronics | Amplifier and circuit noise | **Gaussian** (CLT applied to many micro-disturbances) |
| ADC | Rounding to nearest integer | **Uniform** over one quantization step |

The total reading is the sum of all of these, and by the CLT the sum is **approximately Gaussian** at moderate-to-high signal levels — which is why classical denoising and estimation tools assume Gaussian noise and get away with it.

---

## Worked Example — The Camera Sensor

The camera is the cleanest instance of the template because each stage is a separable piece of silicon you can point at.

### The Full Noise Budget

| Noise Source | Physical Origin | Distribution | Parameters |
|---|---|---|---|
| **Shot noise** | Photon arrival is quantum-random | Poisson($\lambda$) | $\lambda$ = expected photons |
| **Dark current** | Thermal electrons (no light needed) | Poisson($\lambda_d$) | $\lambda_d$ ∝ temperature × time |
| **Read noise** | Amplifier + ADC electronics | Gaussian($0, \sigma_r^2$) | $\sigma_r$ ≈ 1–5 electrons (modern sensors) |
| **Quantization noise** | ADC rounding to integers | Uniform($-\Delta/2, \Delta/2$) | $\Delta$ = full well / $2^{\text{bits}}$ |

**Total signal in electrons:**

$$S = \underbrace{\text{Poisson}(\lambda)}_{\text{photons}} + \underbrace{\text{Poisson}(\lambda_d)}_{\text{dark current}} + \underbrace{\mathcal{N}(0, \sigma_r^2)}_{\text{read noise}} + \underbrace{\text{Uniform}(-\Delta/2, \Delta/2)}_{\text{quantization}}$$

By the CLT, this sum of independent random variables approaches Gaussian with:
- Mean: $\mu = \lambda + \lambda_d$
- Variance: $\sigma^2 = \lambda + \lambda_d + \sigma_r^2 + \Delta^2/12$

### The Complete Signal Chain (Simulation Function)

The full sensor simulation builds a pixel value from first principles:

1. **Photon arrival** — draw from `Poisson(λ)`. This is the quantum noise floor.
2. **Quantum efficiency** — thin each photon count through `Binomial(n_photons, QE)`. Not every photon produces a signal.
3. **Dark current** — add `Poisson(λ_d)` electrons. These accumulate even with no light.
4. **Read noise** — add `Normal(0, σ_r)`. Electronic noise from the readout circuit.
5. **Clipping** — clip to $[0, \text{full well capacity}]$. The sensor cannot count more than its capacity.
6. **ADC quantization** — scale to $[0, 2^{\text{bit depth}} - 1]$ and round to integers.

Every link in this chain is one of the distributions from earlier parts. There is nothing new here — the entire camera noise model is just the four distributions glued together.

### Key observations from the signal chain

1. Shot noise (Poisson) dominates at moderate-to-high photon counts
2. QE reduces the signal but preserves the Poisson shape
3. Dark current adds a small extra Poisson contribution
4. Read noise barely changes the shape (small $\sigma_r$ relative to shot noise at most exposures)
5. The total is well-approximated by Gaussian (by the CLT)
6. ADC quantisation creates discrete steps but doesn't change the envelope

### The Three Regimes of Sensor Noise

Depending on the signal level, different noise sources dominate:

| Regime | Signal Level | Dominant Noise | SNR Behavior |
|---|---|---|---|
| **Read-noise limited** | Dark pixels | Electronics ($\sigma_r$) | SNR ∝ signal (linear) |
| **Shot-noise limited** | Mid-tones | Physics ($\sqrt{\lambda}$) | SNR ∝ $\sqrt{\text{signal}}$ |
| **Saturation** | Bright pixels | Well is full | SNR collapses |

Knowing the regime tells you what denoising strategy applies:
- **Read-noise regime** — averaging frames helps (reduce $\sigma_r$ by $1/\sqrt{n}$)
- **Shot-noise regime** — use signal-dependent denoising (Anscombe transform + Gaussian denoiser)
- **Saturation regime** — information is irreversibly lost; no algorithm recovers it

### Visualizing the Chain — A Brightness Gradient

Simulate a left-to-right brightness ramp through the full chain:
- **Left edge** — grainy (read-noise limited, low SNR)
- **Middle** — smooth Poisson shot noise
- **Right edge** — saturates to white

That is exactly the image a denoising algorithm like NLM, BM3D, or a learned denoiser receives as input. Every classical noise model in image processing is built on top of the chain shown above.

---

## The Same Template — Other Sensors

Once you've internalised the camera example, every other sensor falls into the same template with different physics in each slot.

### Vibration Sensor (Accelerometer)

| Stage | Camera | Vibration |
|---|---|---|
| Physical event | Photon arrivals (Poisson) | Mechanical impacts / shocks (Poisson when the bearing is faulty) |
| Transduction | Quantum efficiency (Binomial thinning) | Piezoelectric coupling efficiency |
| Accumulation | Photons over exposure | Acceleration over a sample interval |
| Electronics | Amplifier + read noise (Gaussian) | Charge amplifier + ADC noise (Gaussian) |
| ADC | $2^{\text{bits}}$ levels | $2^{\text{bits}}$ levels |

The "shock pulses per hour" Poisson model from Part 3 sits at the top. The "silent sensor" Gaussian noise floor from Part 4 sits at the bottom. The integrated reading is a sum, and by the CLT it's Gaussian once the count is high enough.

### Network Telemetry

| Stage | Network |
|---|---|
| Physical event | Packet drops (Poisson, when the link is healthy) |
| Transduction | Probability the drop is observed by the monitor |
| Accumulation | Drops counted per minute |
| Electronics | (none) |
| ADC | (none — already a count) |

No analog stage, but the Poisson → Normal transition still happens as the count per window grows.

### Audio (Microphone)

| Stage | Audio |
|---|---|
| Physical event | Air-pressure variations (continuous, modelled as Gaussian for noise floor) |
| Transduction | Diaphragm coupling |
| Accumulation | Sample-and-hold over the sample period |
| Electronics | Pre-amp + ADC noise (Gaussian) |
| ADC | 16/24-bit quantization (Uniform per step) |

Audio skips the Poisson stage at typical sound levels — but at very low levels (dark-room recording, sensitive astronomical instruments) the photon-equivalent quantum limit reappears.

---

## Key Takeaways

1. **Bernoulli → Binomial → Poisson → Normal** is a chain of increasing abstraction. Each distribution is a limiting case of the one before it, and each simplification makes the math easier while preserving the essential statistics.

2. **Rare-event counting is Poisson — across all domains.** Photon arrivals, bearing shock pulses, packet drops, defect counts. The variance-equals-mean property ($\sigma^2 = \lambda$) means noise is signal-dependent: high-rate measurements are *relatively* cleaner than low-rate ones.

3. **The CLT is why Gaussian assumptions work everywhere.** Aggregated readings, smoothed signals, calibration averages, mini-batch means — all of them converge to Normal regardless of where they started.

4. **Every classical signal-processing algorithm that assumes Gaussian noise is implicitly relying on this entire chain.** Wiener filtering, Kalman filtering, least-squares estimation, learned denoisers — they all sit on top of Bernoulli → Binomial → Poisson → Normal.

5. **The three regimes — electronics-limited, physics-limited, saturated — exist in every sensor.** The names change (read noise vs. amplifier noise vs. quantization noise) but the structure is identical, and so is the prescription for what to do in each regime.

The next part (`applied_sensors.md`) walks through the full chain numerically for the camera sensor as the worked capstone — and shows where the vibration and audio sensors plug into the same framework.
