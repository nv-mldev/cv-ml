# Chapter 12 — Time Series Analysis

## The Question

> "Has birth weight changed over the years? Is there a trend?"

All previous chapters treated observations as unordered. But NSFG records
the **date** of each pregnancy. When order matters — when today's value
depends on yesterday's — we have a **time series**.

---

## What is a Time Series?

A sequence of measurements indexed by time:

$$y_1, y_2, \ldots, y_T$$

where the subscript is a time index. Examples:
- Monthly average birth weight (1990–2002)
- Weekly pregnancy count
- Annual preterm birth rate

**Key difference from cross-sectional data:** observations are not independent.
What happened last month affects this month. Standard statistical methods that
assume independence give wrong answers for time series.

---

## Importing and Cleaning

NSFG's `datend` variable encodes the date of each pregnancy outcome.
We convert it to a year and aggregate by year.

---

## Linear Regression on Time

The simplest time series model: fit a line where $x$ is time and $y$ is the outcome.

$$\hat{y}_t = \alpha + \beta t$$

This models a **linear trend** — a consistent upward or downward drift.

For birth weight over 1990–2002: is $\hat{\beta}$ positive (weights increasing)?
negative (decreasing)? near zero (stable)?

---

## Moving Averages

A **moving average** smooths a time series by averaging nearby points:

$$\text{MA}_k(t) = \frac{1}{k}\sum_{i=0}^{k-1} y_{t-i}$$

where $k$ is the **window size**. Larger windows are smoother but less responsive to changes.

**Why use it?** Real data is noisy. A moving average reveals the underlying trend
by averaging out random fluctuations.

This is also called a **low-pass filter** — it passes slow (long-term) changes
and blocks fast (short-term) noise.

---

## Serial Correlation

Standard statistics assume observations are independent. In time series, they're not.

**Serial correlation** (also: autocorrelation) measures how correlated each observation
is with the observations before it:

$$\text{Corr}(y_t, y_{t-k})$$

where $k$ is the **lag**. If $k=1$, we're asking: does a high value this month
predict a high value next month?

For birth weight by year: serial correlation might be low (birth weight fluctuates
year-to-year without strong persistence). For economic data, serial correlation is
often very high (GDP this quarter predicts next quarter).

---

## Autocorrelation Function (ACF)

The **ACF** plots serial correlation for all lags:

$$\text{ACF}(k) = \text{Corr}(y_t, y_{t-k})$$

- ACF decays quickly → **stationary** series (no trend, no persistence)
- ACF decays slowly → **non-stationary** (strong trend or long memory)
- ACF shows periodic pattern → **seasonal** effects

---

## Missing Values

Real time series have gaps. In NSFG, some years have too few observations
for reliable estimates. Options:

1. **Remove** sparse years
2. **Interpolate** using neighboring years
3. **Model** the missing-data mechanism explicitly

We take option 1 for simplicity.

---

## Prediction

Given a fitted trend line, we can **predict** future values:

$$\hat{y}_{T+h} = \alpha + \beta(T + h)$$

But time series prediction has a fundamental limitation: the further ahead you
predict, the more uncertainty accumulates. Always report prediction intervals,
not just point estimates.

---

## Exercises

1. Aggregate mean birth weight by year. Plot the time series.
2. Fit a linear trend. Is birth weight increasing or decreasing over 1990–2002?
3. Compute a 3-year moving average and overlay it on the original series.
4. Compute and plot the ACF for birth weight by year. Is there serial correlation?
5. Predict the mean birth weight for 2005 using the linear trend. What is your uncertainty?

---

## Glossary

**time series** — a sequence of observations indexed by time

**trend** — a long-term increase or decrease in the series

**moving average** — smoothed series computed as the average of a sliding window

**serial correlation** — correlation between an observation and a lagged version of itself

**ACF (Autocorrelation Function)** — serial correlation plotted as a function of lag

**lag** — the time offset used in serial correlation (lag $k$ → correlation with $k$ steps back)

**stationary** — a time series with constant mean and variance (no trend or seasonality)

**interpolation** — estimating missing values from surrounding observed values

**prediction interval** — a range expected to contain a future observation with stated probability
