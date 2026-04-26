# Chapter 13 — Survival Analysis

## The Question

> "How long until a woman has her next pregnancy?"

This is a different kind of question. It's about **time until an event** — and some
women in the dataset haven't had the event yet. They are still "waiting."

This is the survival analysis problem, and it requires special methods.

---

## The Problem with Ignoring Censoring

Suppose we just take the mean time between pregnancies for women who had two or more.
This ignores the women who only had one pregnancy by the time of the survey.

But those women might go on to have more pregnancies later — we just don't know yet.
They are **censored**: we know they survived past time $t$, but we don't know when (or if) the event occurs.

Ignoring censored observations biases the estimate downward — we only use the
fast pregnancies and throw away information about the slow ones.

---

## Survival Function

The **survival function** $S(t)$ is the probability of surviving past time $t$ —
i.e., the event has NOT occurred by time $t$:

$$S(t) = P(T > t) = 1 - F(t)$$

where $F(t)$ is the CDF of event times.

For inter-pregnancy intervals:
- $S(12) = 0.6$ means 60% of women have NOT had their next pregnancy within 12 months

The survival function always starts at $S(0) = 1$ and decreases toward 0.

---

## Hazard Function

The **hazard function** $h(t)$ is the instantaneous rate of the event occurring
at time $t$, given that it hasn't occurred yet:

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t \mid T \geq t)}{\Delta t} = \frac{f(t)}{S(t)}$$

Intuitively: if you've survived to time $t$, how likely are you to have the event
in the next small interval?

For pregnancies: the hazard might peak at certain months (most common inter-pregnancy intervals).

---

## Kaplan-Meier Estimator

The **Kaplan-Meier (KM) estimator** computes the survival function from data,
correctly handling censored observations.

**Algorithm:**

At each event time $t_j$ (when someone's event occurs):
$$\hat{S}(t_j) = \hat{S}(t_{j-1}) \cdot \left(1 - \frac{d_j}{n_j}\right)$$

where:
- $d_j$ = number of events at time $t_j$
- $n_j$ = number still at risk just before $t_j$ (neither had event nor censored yet)

The key: censored observations contribute to $n_j$ for all times before their censoring,
but don't contribute to $d_j$ — they reduce the "at risk" count but aren't counted as events.

---

## The Marriage Curve — A Related Example

Downey uses time-to-marriage from NSFG as another example of survival analysis.
The survival curve there shows: what fraction of women are still unmarried at each age?

This is the same structure:
- Event = marriage
- Censored = women not yet married at time of survey

The KM estimator handles both identically.

---

## Cohort Effects

Different generations may have different survival curves. Women born in 1960 vs 1975
may have different inter-pregnancy intervals (changed fertility patterns, economics, etc.).

By computing separate KM curves for cohorts and comparing, we can detect these shifts.

---

## Expected Remaining Lifetime

Given that you've survived to time $t$, how much longer do you expect to wait?

$$E[T - t \mid T > t] = \frac{\int_t^\infty S(u)\, du}{S(t)}$$

For inter-pregnancy intervals: given that 12 months have passed without a pregnancy,
how much longer does a woman expect to wait on average?

This answers a different question than the unconditional mean — and the answer differs.

---

## Exercises

1. Using NSFG, compute the inter-pregnancy interval (months between consecutive pregnancies per woman).
2. Identify which women are censored (only one pregnancy recorded).
3. Implement the Kaplan-Meier estimator from scratch.
4. Plot the KM survival curve for inter-pregnancy intervals.
5. Compare KM curves for women in different age cohorts.

---

## Glossary

**survival analysis** — methods for analyzing time-to-event data with possible censoring

**survival function** $S(t)$ — probability of no event before time $t$; $S(t) = 1 - F(t)$

**hazard function** $h(t)$ — instantaneous event rate at time $t$ given survival to $t$

**censored observation** — observation where the event has not (yet) occurred; we know $T > t$ but not $T$

**right censoring** — the most common type: the event hasn't happened by the end of observation

**Kaplan-Meier estimator** — nonparametric estimate of survival function that handles censoring correctly

**at risk** — observations that have not yet had the event and have not been censored

**cohort effect** — differences in survival between groups defined by birth year or entry time
