# Probability for Computer Vision

A ground-up probability series for CV practitioners. The distribution chain —
Bernoulli → Binomial → Poisson → Normal → CLT — is built step by step,
with every concept anchored in a concrete sensor or image-processing
scenario.

## Parts

| File pair | Description |
|-----------|-------------|
| [`part0_what_is_a_distribution`](part0_what_is_a_distribution.md) | Random process, sample space, events, random variables; why averages are not enough; PMF vs PDF; histogram convergence; parameter compression. |
| [`part1_bernoulli`](part1_bernoulli.md) | The Bernoulli trial as the atom of randomness; quantum efficiency (QE) as a Bernoulli probability. |
| [`part2_binomial`](part2_binomial.md) | Counting successes in $n$ trials; PMF built term by term; shape as a function of $n$ and $p$; Monte Carlo validation. |
| [`part3_poisson`](part3_poisson.md) | Binomial limit as $n \to \infty$, $p \to 0$; shot noise derivation; $\sigma = \sqrt{\lambda}$ property; three noise regimes introduced. |
| [`part4_normal`](part4_normal.md) | The bell curve and its two parameters; Poisson → Normal convergence as $\lambda$ grows. |
| [`part5_clt`](part5_clt.md) | Central Limit Theorem: any distribution becomes Gaussian when summed; KS-distance convergence rate. |
| [`part6_putting_it_together`](part6_putting_it_together.md) | Full sensor simulation (photons → ADC); signal chain visualization; three noise regimes; gradient-image capture — the CV capstone. |
| [`exercises`](exercises.md) | Four practice problems: Binomial/Poisson convergence rate, Anscombe transform, noise budget, CLT skewness. |

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

- `cv-ml/book/part4_the_math/ch07_probability/ch07_probability.md` — cites
  this as the simulation backing for the probability chapter.
- `nn-basics/fundamentals/math_concepts.ipynb` — links to specific parts
  from §1 (random variables), §2 (Gaussian distribution), §5 (CLT),
  §6 (law of large numbers).

If you add a new downstream reference, list it here and link back to this
directory so the cross-reference graph stays discoverable.
