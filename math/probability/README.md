# Probability for Signal Processing

A ground-up probability series for engineers and practitioners working with
real-world signals — vibration sensors, audio streams, network telemetry,
camera pixels, time series. The distribution chain —
**Bernoulli → Binomial → Poisson → Normal → CLT** — is built step by step,
with every part led by a concrete applied scenario (a vibration sample
crossing an alarm threshold, bearing-fault counts per hour, a moving-average
filter on a noisy signal) before the math arrives. The camera-sensor noise
model is then worked through end-to-end as the cleanest physical instance
of the chain — every link maps to a separate piece of silicon — and other
sensors are shown to plug into the same template.

## Parts

| File pair | Description |
|-----------|-------------|
| [`part0_what_is_a_distribution`](part0_what_is_a_distribution.md) | The full conceptual ladder — random process → sample space → event → random variable → distribution → parameters — built with two running examples (a synthetic word source and a vibration sensor stream). Signal-generic, stays in $\mathbb{R}^1$. |
| [`part1_bernoulli`](part1_bernoulli.md) | One vibration sample crosses the alarm threshold; the Bernoulli trial as the atom of randomness across signal-processing domains. |
| [`part2_binomial`](part2_binomial.md) | Counting threshold crossings in a 1-second vibration window; PMF built term by term; shape as a function of $n$ and $p$; Monte Carlo validation. |
| [`part3_poisson`](part3_poisson.md) | Bearing shock pulses per hour as the canonical rare-event count; Binomial limit as $n \to \infty$, $p \to 0$; the $\sigma = \sqrt{\lambda}$ rule. |
| [`part4_normal`](part4_normal.md) | The silent-sensor noise floor as the entry point to the bell curve; two parameters, Poisson → Normal convergence as $\lambda$ grows. |
| [`part5_clt`](part5_clt.md) | Moving-average filtering on a noisy stream as the entry point to the CLT; any distribution becomes Gaussian when summed; $1/\sqrt{n}$ noise reduction; KS-distance convergence rate. |
| [`part6_putting_it_together`](part6_putting_it_together.md) | The generic measurement chain (transduction → accumulation → electronics → ADC) worked end-to-end for the camera sensor; vibration / audio / network sensors plug into the same template. |
| [`exercises`](exercises.md) | Four practice problems: Binomial/Poisson convergence rate, Anscombe transform, noise budget, CLT skewness. |
| [`aside_high_dim_distributions`](aside_high_dim_distributions.md) | **Forward-reference, optional.** High-dimensional distributions, manifolds, what neural networks learn in feature space, distribution shift, adversarial examples, generative models. Read after parts 1–6; will be relocated to a Part IV chapter once the CNN material is in place. |
| [`stochastic_processes`](stochastic_processes.md) | **Placeholder.** Time-indexed randomness — stationarity, autocorrelation, power spectral density, ergodicity, named processes (white noise, Wiener, Markov, Poisson). Outline only; will be filled in when Part II (Signals and Measurement) needs it. |

## Running

Every `.py` file is standalone:

```bash
# from project root
source .venv/bin/activate
python math/probability/part0_what_is_a_distribution.py
python math/probability/part1_bernoulli.py
python math/probability/part2_binomial.py
python math/probability/part3_poisson.py
python math/probability/part4_normal.py
python math/probability/part5_clt.py
python math/probability/part6_putting_it_together.py
python math/probability/exercises.py   # stub — complete the exercises first
```

## Who links here

- [`applied_sensors.md`](applied_sensors.md) — the applied capstone of
  this series; uses the Bernoulli → Binomial → Poisson → Normal → CLT
  chain to derive the sensor noise model.
- `nn-basics/fundamentals/math_concepts.ipynb` — links to specific parts
  from §1 (random variables), §2 (Gaussian distribution), §5 (CLT),
  §6 (law of large numbers).

If you add a new downstream reference, list it here and link back to this
directory so the cross-reference graph stays discoverable.
