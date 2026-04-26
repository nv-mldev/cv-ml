# Chapter 8 — Estimation

## The Question

> "We measured 9,148 babies. But we want to say something about ALL babies. How good is our estimate?"

Everything computed so far is a **sample statistic** — a number computed from the data we have.
What we really want is a **population parameter** — the true value in the real world.

The sample mean $\bar{x}$ is our best guess for the population mean $\mu$.
But how far off is it likely to be?

---

## The Estimation Game

Suppose you know a population has a normal distribution with unknown $\mu$ and $\sigma$.
You draw a random sample of $n$ observations. What is your best estimate of $\mu$?

**Answer:** the sample mean $\bar{x}$. This is the **maximum likelihood estimator** of $\mu$
for a normal distribution.

But there's a subtlety for variance. The naive estimate is:

$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

This is **biased** — it systematically underestimates the true $\sigma^2$.
The unbiased estimator divides by $n - 1$ instead:

$$s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$$

**Why $n-1$?** Because we estimated $\bar{x}$ from the data. That used up one
"degree of freedom" — the deviations $(x_i - \bar{x})$ sum to zero, so the last
one is not free. Dividing by $n-1$ corrects for this.

For large $n$, the difference is negligible. For small $n$ (say $n = 5$), it matters.

---

## Sampling Distributions

If you draw many samples and compute the mean of each, the distribution of those
means is called the **sampling distribution of the mean**.

**Key result:** for a population with mean $\mu$ and std $\sigma$,
the sampling distribution of the mean has:

$$E[\bar{X}] = \mu \quad \text{(unbiased)}$$
$$\text{Std}(\bar{X}) = \frac{\sigma}{\sqrt{n}} \quad \text{(standard error)}$$

The standard error shrinks as $\sqrt{n}$ — doubling sample size cuts error by $\sqrt{2}$.

---

## Standard Error

The **standard error (SE)** measures how variable your estimate is:

$$SE = \frac{\sigma}{\sqrt{n}}$$

For birth weight: $\sigma \approx 1.4$ lbs, $n = 9148$, so $SE \approx 0.015$ lbs.
Our estimate of the mean is likely within 0.015 lbs of the true mean.

For the pregnancy length difference (13 hours), we need to compute SE for a difference
of means — this is what determines whether 13 hours is detectable or noise.

---

## Sampling Bias

Not all estimation errors are random. **Bias** is a systematic error that doesn't
average away with more data.

**Example:** the class size paradox from Chapter 3. If you sample students and ask
their class size, you oversample large classes. The estimate of mean class size is
biased upward — taking more data makes it more precise but not more accurate.

**In NSFG:** the dataset uses **survey weights** (`finalwgt`) to correct for
deliberate oversampling of minority groups. If you ignore the weights, your
estimates of national statistics will be biased.

---

## Bootstrap: Simulation-Based Standard Errors

We often can't compute SE analytically. The **bootstrap** gives us a simulation-based alternative:

1. Draw $B$ bootstrap samples: resample with replacement from your data
2. Compute the statistic (mean, median, correlation, anything) for each sample
3. The standard deviation of the $B$ estimates is the bootstrap SE

```python
def bootstrap_mean(data, n_boot=1000):
    estimates = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        estimates.append(sample.mean())
    return np.std(estimates)
```

The bootstrap works because resampling with replacement simulates what would happen
if you repeated the study with new data from the same population.

---

## Exponential Distribution Estimation

For an exponential distribution, the MLE of $\lambda$ is $\hat{\lambda} = 1/\bar{x}$.

But this estimator is biased! The unbiased estimator is:

$$\hat{\lambda} = \frac{n-1}{\sum x_i}$$

The bias disappears as $n \to \infty$, but for small samples it matters.

---

## Exercises

1. Simulate drawing 100 samples of size $n=50$ from birth weight. Plot the sampling distribution of the mean.
2. Compute the standard error of the mean analytically and compare to the simulated std of sample means.
3. Bootstrap the median birth weight. What is the bootstrap SE?
4. Compute the mean with and without survey weights (`finalwgt`). How different are they?
5. Show that the $1/n$ variance estimator is biased: simulate many samples, compute $1/n$ variance each time, average. Does it equal the true variance?

---

## Glossary

**sample statistic** — a value computed from data (e.g., sample mean $\bar{x}$)

**population parameter** — the true value in the underlying population (e.g., $\mu$)

**estimator** — a formula for computing a population parameter from sample data

**bias** — systematic error; $\text{Bias} = E[\hat{\theta}] - \theta$

**unbiased estimator** — $E[\hat{\theta}] = \theta$; right on average

**sampling distribution** — distribution of a statistic over many repeated samples

**standard error (SE)** — standard deviation of the sampling distribution; $SE = \sigma/\sqrt{n}$

**bootstrap** — resample with replacement to simulate the sampling distribution

**degrees of freedom** — number of values free to vary; affects bias in variance estimation
