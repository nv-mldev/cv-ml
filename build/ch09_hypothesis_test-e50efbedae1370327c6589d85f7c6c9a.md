# Chapter 9 — Hypothesis Testing

## The Question

> "First babies are born 13 hours later. But could that just be random noise?"

We have a difference. We want to know: **is this difference real, or is it the kind
of thing that happens by chance even when there is no true effect?**

This is hypothesis testing.

---

## The Logic

We assume, for the sake of argument, that there is no real effect — that first babies
and other babies have the same pregnancy length distribution. This assumption is
called the **null hypothesis** ($H_0$).

If $H_0$ is true, we can simulate what differences we'd expect by chance.
If our observed difference is unlikely under $H_0$, we reject $H_0$.

**The key question:** under the null hypothesis, how often would we see a difference
at least as large as the one we actually observed?

That probability is the **p-value**.

---

## Classical Hypothesis Testing (historical background)

The classical approach (Fisher, Neyman-Pearson):
1. Assume $H_0$: the two groups have the same distribution
2. Compute a test statistic (e.g., difference in means)
3. Compute the p-value: probability of seeing this statistic (or more extreme) under $H_0$
4. If p < 0.05 (arbitrary threshold), "reject $H_0$"

**Problems with classical testing:**
- The 0.05 threshold is arbitrary (Fisher himself said so)
- p < 0.05 does not mean the effect is large or important (Chapter 2 showed d = 0.029)
- p > 0.05 does not mean the null is true — only that we don't have enough evidence
- Multiple comparisons: test 20 things at p = 0.05, one will be significant by chance

---

## The Permutation Test — Build Intuition First

Before any formulas, the cleanest way to understand hypothesis testing is simulation.

**The idea:** if there is no real difference between first and other babies, then which
baby is "first" is arbitrary — just a label. We could shuffle the labels and the
difference in means should be about the same.

**Algorithm:**
1. Pool all observations into one group
2. Randomly split into two groups of the original sizes
3. Compute the difference in means
4. Repeat 1000 times
5. Count how often the simulated difference ≥ the observed difference

That count / 1000 is the **permutation p-value**.

```python
def permutation_test(group1, group2, n_permutations=1000):
    observed = group1.mean() - group2.mean()
    pooled = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        simulated = pooled[:n1].mean() - pooled[n1:].mean()
        if abs(simulated) >= abs(observed):
            count += 1
    return count / n_permutations
```

---

## Testing a Difference in Means

For pregnancy length:
- Observed difference: 0.078 weeks
- Permutation p-value: very small (< 0.001)

Wait — the p-value is tiny, but Chapter 2 showed Cohen's d = 0.029 (tiny effect).
**How can the effect be tiny but the p-value be tiny too?**

Because p-value depends on **sample size**. With n = 9,000, we have enough data
to detect even trivial effects. A statistically significant result is not
necessarily an important result.

**The lesson:** always report effect size AND p-value. Never just one.

---

## Other Test Statistics

The permutation test works with any test statistic, not just the difference in means.

For pregnancy length, we might also use:
- Difference in medians (more robust to outliers)
- Maximum difference in CDFs (Kolmogorov-Smirnov statistic)
- Cohen's d

Each asks a slightly different question. The choice depends on what matters for the problem.

---

## Chi-Squared Test

For categorical data (like birth order), we use the **chi-squared test**.

Example: is the distribution of birth orders what we'd expect from random chance?

$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ is the observed count in category $i$ and $E_i$ is the expected count.

Large $\chi^2$ means the observed counts are far from expected — evidence against $H_0$.

---

## Type I and Type II Errors

| Decision | $H_0$ true | $H_0$ false |
|---|---|---|
| Reject $H_0$ | **Type I error** (false positive) | Correct |
| Fail to reject | Correct | **Type II error** (false negative) |

- Type I error rate = $\alpha$ = significance level (usually 0.05)
- Type II error rate = $\beta$ = depends on effect size and n
- **Power** = $1 - \beta$ = probability of detecting a real effect

Increasing n reduces both error rates. Choosing p < 0.01 reduces Type I but increases Type II.

---

## Exercises

1. Run a permutation test for the difference in mean pregnancy length. What is the p-value?
2. Run a permutation test for the difference in **median** pregnancy length. Same conclusion?
3. Run a permutation test for birth weight. Is the first-baby effect stronger or weaker than for pregnancy length?
4. Implement a chi-squared test for the distribution of birth order.
5. Simulate Type I error: if there is truly no effect, how often does a permutation test give p < 0.05?

---

## Glossary

**null hypothesis** ($H_0$) — the assumption that there is no real effect

**alternative hypothesis** ($H_a$) — the claim we are trying to support

**test statistic** — a number computed from data that summarizes the evidence against $H_0$

**p-value** — probability of observing a test statistic at least as extreme under $H_0$

**permutation test** — a simulation-based test that works by shuffling group labels

**significance level** ($\alpha$) — threshold for p-value; usually 0.05

**Type I error** — rejecting $H_0$ when it is true (false positive); rate = $\alpha$

**Type II error** — failing to reject $H_0$ when it is false (false negative); rate = $\beta$

**power** — $1 - \beta$; probability of detecting a true effect

**chi-squared test** — hypothesis test for categorical data based on observed vs expected counts
