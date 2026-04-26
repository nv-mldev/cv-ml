# Part 2: The Binomial Distribution — Counting Successes

## An Applied Scenario — One Second of Vibration Samples

Back to the motor from Part 1. The accelerometer streams 1,000 samples per second, and your threshold rule fires a `1` whenever a sample exceeds +2 g. On a healthy machine, calibration gave you $p = 0.005$ (1 in 200 samples is a transient).

You decide to summarise the stream one second at a time: **count how many `1`s show up in each 1,000-sample window.** That count is the alarm metric the operator sees.

A few obvious questions:

- What count should you *expect* on a healthy machine?
- How much can the count drift from one second to the next without anything actually being wrong?
- If you see a count of 20 in one window, is that worrying — or just normal variation?

You already have all the ingredients. Each window is $n = 1000$ independent Bernoulli trials with $p = 0.005$. The thing you're counting — successes out of $n$ trials — has a name.

---

## Intuition

If a Bernoulli trial is one coin flip, the **Binomial distribution** answers: "If I run $n$ independent trials, each with success probability $p$, how many successes will I get?"

- **Vibration window**: in 1,000 samples, each with probability $p = 0.005$ of crossing the threshold, how many crossings $k$?
- **Manufacturing batch**: if 200 units are produced and each has probability $p = 0.02$ of being defective, how many defects $k$?
- **Photon counting**: if $n$ photons strike a sensor and each has probability $p = \text{QE}$ of being detected, how many electrons $k$?

In every case the structure is identical: $n$ independent Bernoulli trials with the same $p$, count the successes.

---

## The Math

$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the number of ways to choose which $k$ out of $n$ trials succeed.

**Mean:** $E[k] = np$
**Variance:** $\text{Var}(k) = np(1-p)$

Breaking the formula down:

- $\binom{n}{k}$ — how many ways can exactly $k$ trials succeed out of $n$ total?
- $p^k$ — probability that those $k$ trials all succeed
- $(1-p)^{n-k}$ — probability that the remaining $n-k$ trials all fail
- Multiply: total probability of exactly $k$ successes

---

## Back to the Vibration Sensor

For the motor scenario:

$$n = 1000, \quad p = 0.005 \quad\Rightarrow\quad E[k] = 5, \quad \text{Var}(k) = 4.975, \quad \sigma \approx 2.23$$

So on a healthy machine you should see **about 5 crossings per second**, and a typical fluctuation of $\pm 2$ or so. A count of 7 is unremarkable. A count of 20 is roughly $7\sigma$ above the mean — that's not normal variation, that's a state change worth investigating.

This is a real condition-monitoring rule built from one Bernoulli trial and one sum.

---

## Visualizing the Binomial: Shape vs. $n$ and $p$

The shape depends on both parameters.

- **Increasing $n$ (fix $p$):** the distribution widens and becomes more bell-shaped. This is the first hint of the Normal approximation and the CLT (Part 5).
- **Varying $p$ (fix $n$):** the peak shifts. The distribution is most symmetric when $p = 0.5$; it becomes skewed as $p$ approaches $0$ or $1$.

For the motor case, $p = 0.005$ and $n = 1000$ — extremely small $p$, large $n$. The distribution is heavily skewed and concentrated near small counts. That regime — *many trials, each rare* — is exactly where the Binomial morphs into the Poisson distribution (Part 3).

---

## Simulation: Repeated Windows

Simulating many 1-second windows from the same machine is the empirical version of the formula. Each simulated window is one independent set of $n$ Bernoulli trials.

**Why simulate when we have the formula?** Simulation validates the theory and builds intuition. If simulation and formula disagree, one of them is wrong — a powerful debugging technique that survives all the way into deep learning, where closed-form answers stop existing and Monte Carlo is your only tool.

**What to look for:** the simulation histogram should hug the theoretical PMF. Small residual differences shrink as the number of simulated windows increases (also a CLT effect).

---

## Where Else Binomial Counts Appear

| Domain | $n$ | $p$ | Count $k$ |
|---|---|---|---|
| Vibration monitoring | Samples per window | P(threshold crossing) | Crossings per window |
| Quality control | Units per batch | P(defective unit) | Defects per batch |
| A/B testing | Visitors in a bucket | P(conversion) | Conversions per bucket |
| Image thresholding | Pixels in a patch | P(intensity > T) | Bright pixels per patch |
| Photon counting | Incident photons | Quantum efficiency | Detected electrons |

The formula doesn't care what the trial is — only that the trials are independent and share the same $p$. When $p$ varies across trials or trials aren't independent, you need a different model. We'll handle that in later parts.
