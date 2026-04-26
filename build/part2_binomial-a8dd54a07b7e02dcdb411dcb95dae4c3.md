# Part 2: The Binomial Distribution — Counting Successes

## Intuition

If a Bernoulli trial is one coin flip, the **Binomial distribution** answers: "If I flip the coin $n$ times, how many heads will I get?"

**In CV terms:**
- In a 10×10 image patch, if each pixel independently has probability $p=0.3$ of being an edge pixel, how many edge pixels $k$ will we find?
- In feature matching, if we test $n$ descriptor pairs and each has probability $p$ of being a true match, how many true matches $k$ do we expect?
- In sensor terms: $n$ photons arrive, each with probability $p$ of being detected — how many electrons $k$ will we collect?

## The Math

$$P(k \mid n, p) = \binom{n}{k} p^k (1-p)^{n-k}$$

where $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the number of ways to choose which $k$ out of $n$ trials succeed.

**Mean:** $E[k] = np$
**Variance:** $\text{Var}(k) = np(1-p)$

Let's break this formula apart piece by piece:

- $\binom{n}{k}$ — how many ways can exactly $k$ trials succeed out of $n$ total?
- $p^k$ — probability that exactly those $k$ trials all succeed
- $(1-p)^{n-k}$ — probability that all remaining trials fail
- Multiply all three: total probability of getting exactly $k$ successes

---

## Visualizing the Binomial: What Does the Distribution Look Like?

The shape depends on $n$ (number of trials) and $p$ (success probability).

- **Increasing $n$ (fix $p$):** the distribution widens and becomes more bell-shaped. This is the first hint of the Normal approximation and the CLT.
- **Varying $p$ (fix $n$):** the peak shifts. The distribution is most symmetric when $p = 0.5$; it becomes skewed as $p$ approaches 0 or 1.

---

## Simulation: Repeated Exposures

Let's simulate what happens when we take many exposures of the same scene. Each exposure is an independent set of $n$ Bernoulli trials.

**Why simulate when we have the formula?** Simulation validates the theory and builds intuition. If simulation and formula disagree, one of them is wrong — this is a powerful debugging technique.

**What to look for in the simulation vs PMF plot:** the simulation bars should closely hug the theory line. Small residual differences are expected — they shrink as the number of simulations increases (also a CLT effect).
