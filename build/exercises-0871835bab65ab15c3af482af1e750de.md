# Exercises

## Exercise 1: Binomial to Poisson Convergence — Quantitative

**Task:** For $\lambda = 50$, compute the maximum absolute difference between Binomial($n$, $\lambda/n$) and Poisson($\lambda$) PMFs for $n \in \{50, 100, 500, 1000, 5000\}$. Plot the max difference vs $n$ on a log-log scale. What is the convergence rate?

**Hint:** Use `binom.pmf` and `poisson.pmf` from scipy.stats. Evaluate over $k \in [0, 100]$.

**Expected output:** A log-log plot showing approximately $O(1/n)$ convergence.

### Your task

Implement the convergence measurement:
1. For each n, compute both `Binomial(n, λ/n)` and `Poisson(λ)` PMFs over the same k range
2. Take the maximum absolute difference across all k values
3. Collect results and plot on a log-log scale
4. Fit a line to estimate the convergence rate (slope ≈ -1 means O(1/n))

---

## Exercise 2: Variance-Stabilising Transform

**Task:** The Poisson distribution has signal-dependent noise ($\sigma = \sqrt{\lambda}$). The **Anscombe transform** $f(x) = 2\sqrt{x + 3/8}$ approximately stabilises the variance, making it constant regardless of $\lambda$. This transform is used in **fluorescence microscopy and astronomical imaging** before applying Gaussian denoisers — it converts Poisson-distributed data into a form where standard Gaussian denoising pipelines (NLM, BM3D, Wiener) are valid.

1. Generate Poisson samples for $\lambda \in \{5, 50, 500\}$ (10,000 samples each)
2. Apply the Anscombe transform to each set
3. Compare the variance before and after the transform
4. Plot histograms showing how the transform makes all three distributions have similar spread

**Hint:** After the Anscombe transform, the variance should be approximately 1 for all $\lambda$ values.

**Expected output:** Three pairs of histograms (before/after) showing variance stabilisation.

### Your task

Implement the Anscombe variance stabilization:
1. Generate Poisson samples for each λ and compute the raw variance
2. Apply `f(x) = 2√(x + 3/8)` element-wise
3. Compute the variance of the transformed samples — it should be ≈ 1 for all λ
4. Plot raw variance vs λ (should grow as λ) and transformed variance vs λ (should be flat)

---

## Exercise 3: Build Your Own Noise Budget

**Task:** You are building an image denoising pipeline for industrial inspection. Given these sensor specs:

- Scene illumination: 200 photons/µm²/exposure
- Photosite pitch: 3.45 µm (Basler acA1920-40gm)
- Quantum efficiency: 0.65
- Read noise: 4 electrons
- Dark current: 2 electrons (cooled sensor)
- Full well: 11,000 electrons
- Bit depth: 12

1. Compute the expected electron count per pixel
2. Compute each noise source's contribution (variance)
3. Compute the total noise and SNR at different signal levels
4. Determine where Gaussian denoising is valid vs. where Anscombe pre-processing is needed (hint: at what signal level does Poisson ≈ Normal?)
5. Simulate 10,000 exposures and verify your calculations match

**Hint:** Photosite area = pitch², expected electrons = flux × area × QE. Gaussian denoising is valid when $\lambda \gg 1$ (typically $\lambda > 20$).

**Expected output:** A noise budget table, SNR vs. signal level plot, and histogram confirming theoretical predictions.

### Your task

Build the noise budget:
1. Compute expected electron count from photon flux, pixel area, and QE
2. Sum noise variances: shot + read + dark
3. Compute SNR = signal / σ_total
4. Determine the signal level at which shot noise exceeds read noise (the read-noise limit)
5. Simulate a histogram matching the theoretical distribution and plot both

---

## Exercise 4: CLT Convergence for Different Distributions

**Task:** The CLT convergence rate depends on the **skewness** of the original distribution. Distributions with higher skewness need more samples to converge.

1. Generate samples from: Uniform(0,1), Exponential(1), and Pareto(2, 1) — each with increasing skewness
2. For each, compute the sum of $n$ samples for $n \in \{1, 2, 5, 10, 30, 100\}$
3. Measure the KS distance from Gaussian for each case
4. Plot convergence rate vs $n$ for all three distributions

**Hint:** `scipy.stats.kstest(standardised_data, 'norm')` gives the KS statistic.

**Expected output:** Higher-skewness distributions need more terms to converge (Pareto slowest, Uniform fastest).

### Your task

Implement the skewness-vs-convergence analysis:
1. For each distribution, compute skewness analytically or from samples
2. For each n, draw normalized sums and run the KS test against N(0,1)
3. Plot KS distance vs n for all three distributions on one log-log axes
4. Rank the distributions by convergence speed and relate to their skewness values
