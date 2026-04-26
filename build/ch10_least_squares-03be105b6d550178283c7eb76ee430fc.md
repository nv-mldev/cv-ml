# Chapter 10 — Linear Least Squares

## The Question

> "If I know how long a pregnancy is, how well can I predict birth weight?"

Correlation (Chapter 7) told us *whether* two variables are related.
Least squares tells us *how* — by fitting the best straight line through the data.

---

## Least Squares Fit

We want to find a line $\hat{y} = \alpha + \beta x$ that best fits the data.
"Best" means minimizing the sum of squared errors:

$$\text{SSE} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2$$

Why squared? Squaring penalizes large errors more than small ones, and it gives
a smooth function we can minimize analytically.

Taking derivatives and setting to zero gives the **normal equations**:

$$\hat{\beta} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = r \cdot \frac{\sigma_Y}{\sigma_X}$$

$$\hat{\alpha} = \bar{y} - \hat{\beta}\bar{x}$$

The line always passes through $(\bar{x}, \bar{y})$.

---

## Interpretation

For predicting birth weight from pregnancy length:

$$\hat{\text{weight}} = \alpha + \beta \cdot \text{prglngth}$$

- $\beta$ is the **slope**: each additional week of pregnancy adds $\beta$ pounds
- $\alpha$ is the **intercept**: predicted weight at 0 weeks (not meaningful here — extrapolation)

The intercept is only meaningful within the range of observed data. Extrapolating
a linear model far outside the data range is almost always wrong.

---

## Residuals

A **residual** is the difference between the observed value and the predicted value:

$$e_i = y_i - \hat{y}_i$$

A good fit has residuals that:
1. Are centered at zero (no systematic bias)
2. Are roughly normal (for inference to work)
3. Have constant variance across the range of $x$ (homoscedasticity)
4. Show no pattern when plotted against $x$ (if they do, the relationship is nonlinear)

**Residual plot** is the key diagnostic tool. Always plot residuals vs fitted values.

---

## Goodness of Fit — $R^2$

The **coefficient of determination** $R^2$ measures what fraction of the variance
in $y$ is explained by the model:

$$R^2 = 1 - \frac{\text{SSE}}{\text{SST}}$$

where $\text{SST} = \sum(y_i - \bar{y})^2$ is the total variance (a baseline with no predictor).

- $R^2 = 1$: perfect fit, model explains all variation
- $R^2 = 0$: model is no better than predicting $\bar{y}$ for everyone
- $R^2 = r^2$ for simple linear regression (note: lowercase $r$ from Chapter 7)

For pregnancy length → birth weight: $R^2 \approx 0.19$ — not great.
Most birth weight variance is not explained by pregnancy length alone.

---

## Bootstrap Confidence Intervals for the Slope

Is $\hat{\beta}$ statistically meaningful, or could it be near zero by chance?

Instead of a formula, we use the bootstrap (Chapter 8):
1. Resample the data with replacement
2. Fit the line to the resample
3. Repeat 1000 times
4. The 95% confidence interval is the 2.5th to 97.5th percentile of bootstrap slopes

This makes no assumptions about the distribution of residuals.

---

## Weighted Resampling

NSFG uses survey weights (`finalwgt`). For nationally representative estimates,
we should sample each observation in proportion to its weight.

In bootstrap resampling: instead of uniform resampling, use the weights as
sampling probabilities. This is called **weighted bootstrap**.

---

## Exercises

1. Fit a line predicting birth weight from pregnancy length. Report $\hat{\alpha}$, $\hat{\beta}$, $R^2$.
2. Plot the scatter with the fitted line and residuals.
3. Make a residual plot. Does the residual variance change across pregnancy lengths?
4. Bootstrap the slope 1000 times. What is the 95% confidence interval?
5. Does the slope change substantially when you use survey weights?

---

## Glossary

**least squares** — fitting a line by minimizing the sum of squared errors

**slope** ($\hat{\beta}$) — change in $y$ per unit change in $x$

**intercept** ($\hat{\alpha}$) — predicted $y$ when $x = 0$

**residual** — $e_i = y_i - \hat{y}_i$; the error of each individual prediction

**SSE** — sum of squared errors (residuals); what least squares minimizes

**SST** — total sum of squares; variance of $y$ around its mean

**$R^2$** — fraction of variance explained by the model; $1 - \text{SSE}/\text{SST}$

**homoscedasticity** — constant residual variance across the range of $x$

**residual plot** — plot of residuals vs fitted values; key diagnostic for linear regression

**extrapolation** — using a model outside the range of observed data; almost always unreliable
