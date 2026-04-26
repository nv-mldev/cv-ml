# Chapter 7 — Relationships Between Variables

## The Question

> "Does a mother's age predict the birth weight of her baby?"

All previous chapters studied one variable at a time. Now we ask:
**do two variables move together?** And if so, how strongly?

---

## Scatter Plots

The first tool for any relationship: plot one variable against the other.

For NSFG, plot mother's age (`agepreg`) on the x-axis vs birth weight (`totalwgt_lb`)
on the y-axis. One dot per pregnancy.

**Problem:** with 9,000 points, dots overlap completely. You can't see the pattern.

**Solutions:**
1. **Jitter** — add small random noise to each point so overlapping points spread out
2. **Alpha** — make each dot semi-transparent; dense regions appear darker
3. **Bin and mean** — divide x into bins, plot mean y per bin (shows trend clearly)
4. **Hexbin** — 2D histogram using hexagonal bins

---

## Characterizing Relationships

Before computing a correlation coefficient, ask:

1. Is the relationship **linear** or **curved**?
2. Is it **monotone** (always increasing or always decreasing)?
3. Are there **subgroups** that behave differently?
4. Are there **outliers** that dominate the relationship?

A correlation coefficient summarizes point 1 only. Always look at the scatter plot.

---

## Covariance

Covariance measures how two variables move together:

$$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$

- Positive covariance: when $X$ is high, $Y$ tends to be high
- Negative covariance: when $X$ is high, $Y$ tends to be low
- Zero covariance: no linear relationship

**Problem:** covariance has units (lbs × years for our example).
Comparing covariances across different pairs of variables is meaningless.

---

## Pearson's Correlation

Normalize covariance by the product of standard deviations:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

Now $r \in [-1, 1]$:
- $r = +1$: perfect positive linear relationship
- $r = -1$: perfect negative linear relationship
- $r = 0$: no linear relationship
- $|r| > 0.7$: strong (by convention)

**Critical warning:** $r = 0$ does not mean "no relationship" — it means
no *linear* relationship. The data could have a strong curved relationship with $r = 0$.

---

## Nonlinear Relationships

For mother's age vs birth weight, the relationship is nonlinear:
- Very young mothers (teens) have lower birth weight babies
- Mothers in their 20s and 30s have the heaviest babies
- Very old mothers (40+) have slightly lower weight babies

Pearson's $r$ misses this U-shape. **Always plot before computing $r$.**

---

## Spearman's Rank Correlation

Spearman's $\rho$ is Pearson's $r$ applied to the **ranks** of the data, not the values.

1. Replace each $x_i$ with its rank (1 = smallest, $n$ = largest)
2. Replace each $y_i$ with its rank
3. Compute Pearson's $r$ on the ranks

$$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference in ranks.

**Advantages over Pearson's $r$:**
- Works for any monotone relationship (not just linear)
- Robust to outliers (one extreme value can't dominate)
- Works on ordinal data

**When to use which:**
- Linear relationship, no outliers → Pearson
- Monotone relationship, or outliers present → Spearman

---

## Correlation and Causation

**The most abused concept in statistics.**

Correlation tells you two variables are associated. It says nothing about why.

Ice cream sales and drowning rates are positively correlated.
**Cause:** both increase in summer (confound = season).

In NSFG, older mothers have slightly lighter babies.
**Possible causes:**
- Direct: older maternal age causes lower birth weight
- Confound: older mothers have more complicated pregnancies
- Reverse: nothing — this is an observational study

Establishing causation requires either:
1. A randomized controlled experiment (randomize the cause)
2. A natural experiment (quasi-random assignment occurs naturally)
3. Careful matching/adjustment for confounds

Observational data like NSFG can only establish association.

---

## Exercises

1. Make a scatter plot of mother's age vs birth weight with alpha=0.1. What pattern do you see?
2. Compute Pearson's $r$ between age and birth weight. Is it strong?
3. Compute Spearman's $\rho$ for the same pair. How does it compare to Pearson?
4. Bin mother's age into 5-year groups and plot mean birth weight per bin. What shape do you see?
5. Compute $r$ between pregnancy length and birth weight. Which pair is more strongly correlated?

---

## Glossary

**scatter plot** — plot of one variable vs another; one point per observation

**jitter** — small random noise added to avoid overplotting

**covariance** — $\text{Cov}(X,Y)$; measures direction of linear association; has units

**Pearson's r** — normalized covariance; unit-free; measures linear association

**Spearman's ρ** — Pearson's r on ranks; measures monotone association; robust to outliers

**monotone relationship** — a relationship that is always increasing or always decreasing

**confound** — a third variable that causes both $X$ and $Y$, creating spurious correlation

**correlation ≠ causation** — association between variables does not imply one causes the other
