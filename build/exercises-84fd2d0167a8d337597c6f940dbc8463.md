# Exercises — Across All 14 Chapters

A collection of end-of-chapter exercises plus cross-chapter synthesis problems.
All use the NSFG dataset. Run `python data/download_nsfg.py` first.

---

## Chapter 1 — EDA
1. Load the NSFG data. How many total pregnancies? How many live births?
2. What is the mean age of mothers at the end of their first recorded pregnancy?
3. What fraction of live births are first babies?
4. Plot a histogram of `agepreg` for first-time mothers only. What shape do you see?

## Chapter 2 — Distributions
5. Compute Cohen's d for birth weight (first vs other). Is the effect larger or smaller than for pregnancy length?
6. What fraction of pregnancies last less than 37 weeks (premature)?
7. Plot normalized histograms of birth weight for all three outcomes: live birth, stillbirth, induced abortion. What differences do you see?

## Chapter 3 — PMF
8. Build a PMF of birth order. What is the most common birth order?
9. Implement the size-biased distribution for birth order. How does the mean change?
10. At which pregnancy length do first babies and other babies differ most in probability?

## Chapter 4 — CDF
11. What is the IQR of birth weight for live births?
12. A baby weighs 5.0 lbs. What percentile is this? (Use the CDF.)
13. Generate 1000 synthetic birth weights using the inverse CDF. Do they match the original distribution?

## Chapter 5 — Modeling
14. Fit a normal distribution to pregnancy length. Make a normal probability plot. Is the fit good?
15. Which fits better for birth weight — Normal or Lognormal? Use the KS test to decide.

## Chapter 6 — PDF
16. Compute skewness and kurtosis for birth weight and mother's age.
17. Plot KDE of birth weight with three bandwidths. Which looks right?
18. Implement a Gaussian KDE from scratch (not using scipy). Verify against scipy's output on a sample.

## Chapter 7 — Relationships
19. Compute Pearson's r for pregnancy length vs birth weight. Is it stronger than age vs weight?
20. Plot a scatter of birth order vs birth weight. Is there a trend?
21. Compute Spearman's ρ for mother's age vs pregnancy length.

## Chapter 8 — Estimation
22. Show that the 1/n variance estimator is biased using simulation (n=10, 10,000 repetitions).
23. Bootstrap the median pregnancy length for first babies. What is the 95% CI?
24. Compute mean birth weight with and without survey weights. How large is the difference?

## Chapter 9 — Hypothesis Testing
25. Run a permutation test for birth weight (first vs other). What is the p-value?
26. Run a permutation test for the difference in medians of pregnancy length.
27. Simulate Type I error rate: how often does a permutation test give p < 0.05 when there is truly no effect?
28. At what sample size does the first-baby effect in pregnancy length become statistically significant (p < 0.05)?

## Chapter 10 — Least Squares
29. Fit a line predicting birth weight from pregnancy length. Report slope, intercept, R².
30. Bootstrap the slope 1000 times. What is the 95% CI?
31. Make a residual plot. Is there any pattern?

## Chapter 11 — Regression
32. Fit multiple regression: `totalwgt_lb ~ prglngth + agepreg + birthord`. Which predictor has the largest coefficient?
33. Does adding a quadratic age term improve adjusted R²?
34. Fit logistic regression predicting preterm birth (prglngth < 37) from mother's age. What is the odds ratio for age?

## Chapter 12 — Time Series
35. Aggregate mean birth weight by year. Is there an upward or downward trend?
36. Compute a 3-year moving average and overlay it on the raw series.
37. Compute the ACF of yearly birth weight. Is there meaningful serial correlation?

## Chapter 13 — Survival Analysis
38. Compute inter-pregnancy intervals. What fraction of women are censored?
39. Implement the Kaplan-Meier estimator from scratch. What fraction of women have their next pregnancy within 18 months?
40. How does the naive mean (ignoring censoring) compare to the KM estimate?

## Chapter 14 — Analytic Methods
41. Demonstrate the CLT: show that sampling distributions of the mean converge to normal as n grows.
42. Compare t-test and permutation test p-values for pregnancy length. Do they agree?
43. Compute the analytic 95% CI for mean birth weight. Compare to bootstrap CI.

---

## Synthesis Problems

**S1.** The full pipeline.
Using NSFG:
- Compute a summary statistic of your choice
- Bootstrap its confidence interval
- Run a permutation test
- Compute Cohen's d
- Write one paragraph interpreting all four results together

**S2.** Model comparison.
Fit three models for birth weight:
1. `totalwgt_lb ~ prglngth`
2. `totalwgt_lb ~ prglngth + agepreg`
3. `totalwgt_lb ~ prglngth + agepreg + birthord`

For each: report R², adjusted R², AIC. Which model would you choose and why?

**S3.** The full story.
You now have all the tools to answer: **"Are first babies born later, and does it matter?"**

Write a 200-word analysis (as if reporting to a non-technical audience) that covers:
- The observed difference
- Whether it is statistically real (hypothesis test)
- Whether it is practically important (effect size)
- What other factors might explain it (regression)
- What your conclusion is
