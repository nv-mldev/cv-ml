"""
Chapter 11 — Regression
Question: What combination of variables best predicts birth weight?

What this script builds:
  - Multiple OLS regression using statsmodels
  - Polynomial regression for nonlinear age effect
  - Logistic regression predicting preterm birth
  - Model comparison: adjusted R^2, AIC, BIC

Run: python ch11_regression.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

try:
    import statsmodels.formula.api as smf
except ImportError:
    print("Install statsmodels: pip install statsmodels")
    exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ch01_eda import live

COLORS = {
    'simple':   '#2196F3',
    'multiple': '#4CAF50',
    'logistic': '#F44336',
    'neutral':  '#9E9E9E',
}

# ── Prepare data ──────────────────────────────────────────────────────────────
df = live[['prglngth', 'totalwgt_lb', 'agepreg', 'birthord']].dropna().copy()
df = df[(df['totalwgt_lb'] > 0) & (df['totalwgt_lb'] < 20)]
df = df[(df['prglngth'] >= 27) & (df['prglngth'] <= 44)]
df = df[(df['agepreg'] > 10) & (df['agepreg'] < 50)]

df['agepreg_sq'] = df['agepreg'] ** 2           # quadratic term for age
df['preterm']    = (df['prglngth'] < 37).astype(int)  # binary outcome

# ── Model 1: Simple regression (baseline) ────────────────────────────────────
m1 = smf.ols('totalwgt_lb ~ prglngth', data=df).fit()

# ── Model 2: Multiple regression ─────────────────────────────────────────────
m2 = smf.ols('totalwgt_lb ~ prglngth + agepreg + birthord', data=df).fit()

# ── Model 3: Polynomial age ───────────────────────────────────────────────────
m3 = smf.ols('totalwgt_lb ~ prglngth + agepreg + agepreg_sq + birthord', data=df).fit()

print("── Model Comparison ────────────────────────────────────────────────────")
print(f"  {'Model':<40}  {'R²':>6}  {'Adj R²':>8}  {'AIC':>10}  {'BIC':>10}")
for name, model in [
    ('prglngth only',                        m1),
    ('prglngth + age + birthord',            m2),
    ('prglngth + age + age² + birthord',     m3),
]:
    print(f"  {name:<40}  {model.rsquared:>6.4f}  "
          f"{model.rsquared_adj:>8.4f}  {model.aic:>10.1f}  {model.bic:>10.1f}")

print(f"\n── Multiple Regression Coefficients ────────────────────────────────────")
print(m2.summary().tables[1])

print(f"\n── Key Interpretations ─────────────────────────────────────────────────")
for var, label in [('prglngth', 'Pregnancy length'),
                   ('agepreg',  "Mother's age"),
                   ('birthord', 'Birth order')]:
    coef = m2.params[var]
    pval = m2.pvalues[var]
    sig  = "significant" if pval < 0.05 else "not significant"
    print(f"  {label:<20}: {coef:+.4f} lbs per unit  (p={pval:.4f}, {sig})")

# ── Logistic regression: predicting preterm birth ─────────────────────────────
m_logit = smf.logit('preterm ~ agepreg + birthord', data=df).fit(disp=False)

print(f"\n── Logistic Regression: Predicting Preterm Birth ───────────────────────")
print(f"  Outcome: preterm birth (prglngth < 37 weeks)")
print(f"  Preterm rate in data: {df['preterm'].mean()*100:.1f}%")
print()
for var in ['agepreg', 'birthord']:
    coef = m_logit.params[var]
    odds_ratio = np.exp(coef)
    pval = m_logit.pvalues[var]
    print(f"  {var:<12}: coef={coef:+.4f}, odds ratio={odds_ratio:.4f}, p={pval:.4f}")

print(f"\n  Odds ratio interpretation:")
print(f"  birthord odds ratio = {np.exp(m_logit.params['birthord']):.4f}")
print(f"  Each additional birth order multiplies odds of preterm by this factor.")

# ── Visualisation ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 1. Simple vs multiple regression predictions
ax = axes[0, 0]
prg_range = np.linspace(27, 44, 100)
pred_simple   = m1.params['Intercept'] + m1.params['prglngth'] * prg_range
mean_age  = df['agepreg'].mean()
mean_bord = df['birthord'].mean()
pred_multi = (m2.params['Intercept'] +
              m2.params['prglngth'] * prg_range +
              m2.params['agepreg']  * mean_age +
              m2.params['birthord'] * mean_bord)
ax.scatter(df['prglngth'], df['totalwgt_lb'], alpha=0.05, s=3, color=COLORS['neutral'])
ax.plot(prg_range, pred_simple, color=COLORS['simple'],   linewidth=2, label='Simple')
ax.plot(prg_range, pred_multi,  color=COLORS['multiple'], linewidth=2, linestyle='--', label='Multiple (at mean age/order)')
ax.set_xlabel('Pregnancy length (weeks)')
ax.set_ylabel('Birth weight (lbs)')
ax.set_title('Simple vs Multiple Regression')
ax.legend(fontsize=9)

# 2. Nonlinear age effect
ax = axes[0, 1]
age_range = np.linspace(15, 45, 100)
pred_linear = (m2.params['Intercept'] +
               m2.params['prglngth'] * 39 +
               m2.params['agepreg']  * age_range +
               m2.params['birthord'] * 1)
pred_quad   = (m3.params['Intercept'] +
               m3.params['prglngth']    * 39 +
               m3.params['agepreg']     * age_range +
               m3.params['agepreg_sq']  * age_range**2 +
               m3.params['birthord']    * 1)
# Binned means for reference
bins = np.arange(15, 48, 3)
bin_means, bin_centers = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (df['agepreg'] >= lo) & (df['agepreg'] < hi)
    if mask.sum() > 20:
        bin_means.append(df.loc[mask, 'totalwgt_lb'].mean())
        bin_centers.append((lo + hi) / 2)
ax.plot(bin_centers, bin_means, 'o', color=COLORS['neutral'], markersize=6, label='Binned means')
ax.plot(age_range, pred_linear, color=COLORS['simple'],   linewidth=2, label='Linear age')
ax.plot(age_range, pred_quad,   color=COLORS['multiple'], linewidth=2, linestyle='--', label='Quadratic age')
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Predicted birth weight (lbs)')
ax.set_title('Nonlinear Age Effect on Birth Weight')
ax.legend(fontsize=9)

# 3. Logistic curve: probability of preterm by age
ax = axes[1, 0]
pred_preterm_prob = m_logit.predict(pd.DataFrame({
    'agepreg': age_range,
    'birthord': np.ones(len(age_range))
}))
ax.plot(age_range, pred_preterm_prob * 100, color=COLORS['logistic'], linewidth=2)
ax.axhline(df['preterm'].mean() * 100, color='grey', linestyle='--',
           label=f'Overall rate: {df["preterm"].mean()*100:.1f}%')
ax.set_xlabel("Mother's age (years)")
ax.set_ylabel('Predicted probability of preterm (%)')
ax.set_title('Logistic Regression: Preterm Probability by Age')
ax.legend(fontsize=9)

# 4. Model comparison bar chart
ax = axes[1, 1]
models = ['Simple\n(prglngth)', 'Multiple\n(+age,order)', 'Poly age\n(+age²)']
r2_vals  = [m1.rsquared, m2.rsquared, m3.rsquared]
adj_r2   = [m1.rsquared_adj, m2.rsquared_adj, m3.rsquared_adj]
x_pos = np.arange(len(models))
ax.bar(x_pos - 0.2, r2_vals,  width=0.4, color=COLORS['simple'],   alpha=0.8, label='R²')
ax.bar(x_pos + 0.2, adj_r2,   width=0.4, color=COLORS['multiple'], alpha=0.8, label='Adjusted R²')
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('R²')
ax.set_title('Model Comparison: R² vs Adjusted R²')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('ch11_regression.png', dpi=150)
plt.show()
print("\nFigure saved: ch11_regression.png")
