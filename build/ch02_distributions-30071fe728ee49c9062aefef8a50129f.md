# Chapter 2 — Distributions

## The Question

> "First babies are born 13 hours later on average. Is that a big difference?"

Chapter 1 gave us a number. But a number without context is meaningless.

13 hours sounds large. But if pregnancy length varies by weeks between women,
13 hours might be invisible. To judge whether a difference is large, we need to
understand the **shape of the data** — the distribution.

---

## What is a Distribution?

A **distribution** tells you two things about a variable:
1. What values are possible?
2. How often does each value appear?

The simplest way to represent a distribution is a **histogram**.

---

## Histograms

A histogram divides the range of values into **bins** and counts how many
observations fall in each bin.

```
Bin [38, 39):  ████████████████  1,243 pregnancies
Bin [39, 40):  ██████████████    1,089 pregnancies
Bin [40, 41):  ████████          612 pregnancies
```

**Binning choices matter.** Too few bins → you lose shape. Too many bins → noise
looks like signal. This is a judgment call — always try multiple bin widths.

### The Frequency Trap

If you plot raw counts and your two groups have different sizes, the taller bar
is always the bigger group — not necessarily the more common value. Always
**normalize** before comparing:

$$\text{proportion} = \frac{\text{count in bin}}{\text{total count}}$$

---

## NSFG Variables We Study This Chapter

| Variable | What we ask |
|---|---|
| `prglngth` | Do first babies have a different distribution of pregnancy length? |
| `totalwgt_lb` | How is birth weight distributed? What are the outliers? |
| `agepreg` | What is the age distribution of mothers? |

---

## Outliers

Every real dataset has outliers — values that are far from the rest.

In NSFG `prglngth`, you'll see values of 0 weeks (unclear what this means)
and values above 45 weeks (rare but possible). Before any analysis, you must
decide what to do with them:

1. **Remove them** — if they represent data errors
2. **Keep them** — if they represent genuine rare events
3. **Flag them** — compute results with and without, report both

For pregnancy length, we restrict to 27–44 weeks for live births.
Shorter than 27 weeks is extreme prematurity (rare, different medical category).
Longer than 44 weeks is likely a recording error.

---

## First Babies vs Others — Visually

When you plot the two histograms together, the difference is subtle.
Both distributions are strongly peaked at 39 weeks.
The first-baby distribution is very slightly shifted right.

This is the visual evidence that the 13-hour difference is **small relative
to the spread**. Most first babies are born at the same time as other babies.

---

## Summarizing a Distribution

Sometimes you want a single number instead of a plot. The most common summaries:

### Mean (average)
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

Sensitive to outliers. One very large value pulls the mean far from the center.

### Variance
$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

The average squared deviation from the mean. Units are squared (weeks²).

### Standard Deviation
$$\sigma = \sqrt{\sigma^2}$$

Same units as the data. A typical pregnancy deviates from the mean by $\sigma$ weeks.

---

## Effect Size — Cohen's d

The problem with raw differences: "0.078 weeks" sounds small. But how small?
Is it 1% of the typical variation or 50%?

**Cohen's d** measures the difference in means, normalized by the pooled
standard deviation:

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

where the pooled standard deviation is:

$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

**Rough guidelines (Cohen 1988):**
- $|d| < 0.2$ → small effect
- $|d| \approx 0.5$ → medium effect
- $|d| > 0.8$ → large effect

For the first-baby pregnancy length difference, $d \approx 0.029$ — tiny.
The anecdote is technically true but practically meaningless.

---

## Reporting Results

When you report a statistical result, always report:
1. The **effect size** (not just whether it is significant)
2. The **sample sizes** (larger samples detect smaller effects)
3. The **direction** of the effect

Bad reporting: "First babies are born later (p < 0.05)."
Good reporting: "First babies have a mean pregnancy length 0.078 weeks longer
(d = 0.029, n₁ = 4,413, n₂ = 4,735), a statistically detectable but
practically negligible difference."

---

## The Pivot — Back to Probability

Look at the birth weight histogram. It is roughly bell-shaped, symmetric, peaked near 7.4 lbs.

You've seen this shape before — or you're about to.

**The question the histogram raises:**
> "Is there a mathematical model that perfectly describes this shape?
> One with just 2 numbers — a mean and a spread — that captures everything?"

The answer is the **Normal distribution**. And the reason it appears everywhere is
the **Central Limit Theorem**.

Both of these are in `probability/part4_normal` and `probability/part5_clt`.

**Go there now.** Learn why this shape appears, what math produces it, and what
its parameters mean. Then come back to Chapter 3 — where we use that knowledge
to ask harder questions about the same data.

---

## Exercises

1. Compute mean, variance, and std for `prglngth` for first vs other babies.
2. Compute Cohen's d for pregnancy length. Is it small, medium, or large?
3. Plot histograms of `totalwgt_lb` for first vs other babies. What do you notice?
4. What fraction of pregnancies have `prglngth` < 37 weeks (premature)?
5. Compute Cohen's d for birth weight. Compare it to the pregnancy length effect.

---

## Glossary

**distribution** — the set of possible values of a variable and the frequency of each

**histogram** — a plot that counts how many values fall in each bin

**bin** — an interval used to group values in a histogram

**normalization** — dividing counts by total to get proportions (removes group size effects)

**outlier** — a value far from the rest of the data

**mean** — sum of values divided by count; sensitive to outliers

**variance** — average squared deviation from the mean

**standard deviation** — square root of variance; same units as the data

**Cohen's d** — difference in means divided by pooled std; a unit-free effect size measure

**effect size** — how large a difference is, expressed in meaningful units
