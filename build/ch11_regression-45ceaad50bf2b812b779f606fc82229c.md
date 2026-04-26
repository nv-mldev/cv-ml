# Chapter 11 — Regression

## The Question

> "Pregnancy length partly predicts birth weight. What else does?"

Chapter 10 used one predictor. Real phenomena are shaped by many variables.
**Multiple regression** lets us use all of them simultaneously.

---

## Multiple Regression

We extend the simple model to include multiple predictors:

$$\hat{y} = \alpha + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_k x_k$$

For birth weight we might include:
- $x_1$ = pregnancy length (weeks)
- $x_2$ = mother's age (years)
- $x_3$ = birth order (1st, 2nd, ...)

**Matrix form:**

$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\beta}$$

OLS solution: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

---

## Interpreting Multiple Regression Coefficients

Each coefficient $\hat{\beta}_j$ is the change in $y$ per unit change in $x_j$,
**holding all other variables constant**.

This is critical: in simple regression, the age coefficient captures both
the direct effect of age AND any confounds correlated with age.
In multiple regression, each coefficient is adjusted for the others.

Example: in simple regression, pregnancy length predicts birth weight.
But longer pregnancies and heavier babies might both be correlated with mother's
health. Multiple regression can separate these.

---

## Nonlinear Relationships in Regression

Chapter 7 showed that mother's age has a nonlinear relationship with birth weight —
teens and older mothers have lighter babies, peak in the 30s.

We can model this with a **quadratic term**:

$$\hat{\text{weight}} = \alpha + \beta_1 \cdot \text{age} + \beta_2 \cdot \text{age}^2$$

The model is still linear in the **coefficients** — we just created a new variable
$x_2 = x_1^2$. This is called **polynomial regression**.

---

## Logistic Regression — Binary Outcomes

What if the outcome is binary — e.g., preterm birth (yes/no)?

We can't use linear regression for binary outcomes (it predicts probabilities outside [0,1]).
Instead, we use **logistic regression**, which models the log-odds:

$$\log\frac{p}{1-p} = \alpha + \beta_1 x_1 + \beta_2 x_2$$

Solving for $p$:

$$p = \frac{1}{1 + e^{-(\alpha + \beta_1 x_1 + \beta_2 x_2)}}$$

This is the **sigmoid function** — the same one used in neural networks.

**Coefficients:** $e^{\hat{\beta}_j}$ is the **odds ratio** for variable $j$.

---

## Using statsmodels

We use statsmodels (not sklearn) because it gives us:
- Full statistical output (p-values, confidence intervals, F-statistics)
- The same interface as R — important for academic work
- AIC/BIC for model comparison

```python
import statsmodels.formula.api as smf

model = smf.ols('totalwgt_lb ~ prglngth + agepreg + birthord', data=df).fit()
print(model.summary())
```

---

## Model Comparison

How do we know if adding a predictor improves the model?

1. **$R^2$** increases when you add any variable (even useless noise) — use **adjusted $R^2$**
2. **AIC / BIC** penalize complexity — lower is better; use to compare models
3. **F-test** tests whether a group of coefficients is jointly zero

Adjusted $R^2$:
$$\bar{R}^2 = 1 - \frac{(1 - R^2)(n-1)}{n - k - 1}$$

where $k$ is the number of predictors.

---

## Exercises

1. Fit multiple regression: `totalwgt_lb ~ prglngth + agepreg + birthord`. Report coefficients.
2. Add a quadratic age term (`agepreg**2`). Does adjusted $R^2$ improve?
3. Fit logistic regression predicting preterm birth (prglngth < 37 weeks) from mother's age and birth order.
4. What is the odds ratio for birth order on preterm birth?
5. Compare AIC of the simple model (prglngth only) vs the full model.

---

## Glossary

**multiple regression** — regression with more than one predictor

**coefficient** — $\hat{\beta}_j$: change in outcome per unit change in $x_j$, holding others constant

**polynomial regression** — adding squared/cubic terms to model nonlinear relationships

**logistic regression** — regression for binary outcomes; models log-odds

**sigmoid function** — $\sigma(z) = 1/(1+e^{-z})$; maps any value to (0,1)

**odds ratio** — $e^{\hat{\beta}}$; multiplicative change in odds per unit increase in $x$

**adjusted $R^2$** — $R^2$ penalized for number of predictors; prevents overfitting

**AIC** — Akaike Information Criterion; model quality + complexity penalty; lower = better

**BIC** — Bayesian Information Criterion; heavier complexity penalty than AIC
