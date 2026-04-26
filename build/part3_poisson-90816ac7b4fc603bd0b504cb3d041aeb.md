# Part 3: The Poisson Distribution — The Limit of Many Rare Events

## An Applied Scenario — Bearing Faults Per Hour

A different sensor on the same motor monitors **shock pulses** — the high-frequency clicks a damaged ball bearing emits as it rolls past a load zone. On a healthy bearing, these clicks are extremely rare. On a damaged one, they multiply.

You want a maintenance metric: **how many shock pulses arrive in a one-hour window?**

Try modelling this with the Binomial. You'd need:

- $n$ — the number of "trials" per hour. But what is a trial? A nanosecond? A bearing rotation? It depends on how finely you slice time.
- $p$ — the probability of a click in one trial. As you slice time finer, $n$ grows and $p$ shrinks.

Yet the *expected count per hour* — say, $\lambda = 3$ clicks/hour on a healthy bearing — is a real, measurable physical quantity. It doesn't depend on how you slice time.

You want a distribution parameterised by $\lambda$ alone, where the trial-counting machinery quietly disappears. That distribution exists, and it's what the Binomial becomes in exactly this regime.

---

## Intuition

The Binomial works perfectly when you know $n$ and $p$ separately. But many counting problems involve events that are **rare across a large region** — and the region's "size" is what you actually measure, not $n$ and $p$ individually.

The **Poisson distribution** is what the Binomial becomes when:
- $n \to \infty$ (the region is sliced into infinitely many trials)
- $p \to 0$ (each trial is vanishingly unlikely)
- $\lambda = np$ stays constant (the expected count is fixed by the physics)

The two parameters collapse into one: $\lambda$, the expected count per region.

---

## Examples Across Domains

| Process | Underlying $n$ | Underlying $p$ | $\lambda$ |
|---|---|---|---|
| Bearing shock pulses | Time slices per hour | P(click in slice) | Clicks per hour |
| Network packet drops | Packets per minute | P(drop) | Drops per minute |
| Photon counting | Available photons | P(photon hits photosite) | Expected electrons per exposure |
| Defects on a surface | Surface micro-cells | P(defect per cell) | Defects per m² |
| Earthquakes in a region | Crustal "trial" stress events | P(release per event) | Events per year |
| Background pixel changes | Pixels per frame | P(noise above threshold) | Changed pixels per frame |

In every row: huge $n$, tiny $p$, moderate $\lambda$. The model doesn't care what kind of "event" you're counting.

---

## The Derivation

Start from the Binomial PMF and take the limit.

$$P(k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Substitute $p = \lambda / n$:

$$P(k) = \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

### Step-by-step limit

Expand the binomial coefficient:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}$$

For large $n$, each factor $n, (n-1), (n-2), \ldots$ is approximately $n$:

$$\binom{n}{k} \approx \frac{n^k}{k!}$$

Substitute back:

$$P(k) \approx \frac{n^k}{k!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

The $n^k$ terms cancel:

$$P(k) \approx \frac{\lambda^k}{k!} \cdot \left(1 - \frac{\lambda}{n}\right)^{n} \cdot \left(1 - \frac{\lambda}{n}\right)^{-k}$$

As $n \to \infty$:
- $\left(1 - \frac{\lambda}{n}\right)^{n} \to e^{-\lambda}$ (the definition of $e$)
- $\left(1 - \frac{\lambda}{n}\right)^{-k} \to 1$ (since $k$ is fixed and $\lambda/n \to 0$)

$$\boxed{P(k \mid \lambda) = \frac{\lambda^k \, e^{-\lambda}}{k!}}$$

**Mean:** $E[k] = \lambda$
**Variance:** $\text{Var}(k) = \lambda$

The magical property: **the mean equals the variance**. This single fact is the engine behind every shot-noise calculation in sensors, every queueing model in networks, and every count-based regression in statistics.

---

## Back to the Bearing

For the healthy bearing with $\lambda = 3$ clicks/hour:

- Expected count per hour: $3$
- Standard deviation: $\sqrt{3} \approx 1.73$
- $P(0 \text{ clicks}) = e^{-3} \approx 0.050$
- $P(\geq 10 \text{ clicks}) \approx 0.0011$

So observing 0 clicks in an hour is uncommon but not alarming (5% of healthy hours look like this). Observing 10 clicks in an hour is a 1-in-1000 event under the healthy model — strong evidence the bearing state has changed.

The same logic, with different $\lambda$, runs every condition-monitoring threshold in the building.

---

## Why Poisson is the Right Model for Photon Counting

A typical LED emits ~$10^{18}$ photons per second. The probability that any specific photon reaches a 6 µm × 6 µm photosite is vanishingly small. But the product $\lambda = np$ — determined by illumination, reflectance, exposure time, and sensor area — sits in the range of tens to thousands.

This is exactly the Poisson regime: enormous $n$, tiny $p$, moderate $\lambda$. The Poisson model is not an approximation here — it is the physically correct distribution for photon counting.

---

## The Poisson Has ONE Parameter — Why That Matters

| Distribution | Parameters | Mean | Variance | Mean = Variance? |
|---|---|---|---|---|
| Binomial($n$, $p$) | $n$, $p$ | $np$ | $np(1-p)$ | Only when $p \to 0$ |
| Poisson($\lambda$) | $\lambda$ | $\lambda$ | $\lambda$ | **Always** |

If you measure the mean of a Poisson process, you immediately know its variance. Sensor designers use this constantly — they predict noise from signal level alone. The same trick works for any system in the Poisson regime: a packet-loss monitor, a click-rate model, or a defect-counting pipeline.
