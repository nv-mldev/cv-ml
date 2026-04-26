# Part 1: The Bernoulli Trial — The Atom of Randomness

## Intuition

Every probability distribution we'll study is built from the simplest possible random event: **something either happens or it doesn't.**

- A coin lands heads or tails
- **A pixel either exceeds a threshold or it doesn't** (binarization, Otsu's method)
- **A descriptor distance either falls below a match threshold or it doesn't** (feature matching)
- **A classifier outputs class A or class B** (binary classifier)
- A photon either gets absorbed by the sensor or it doesn't (sensor physics)

This binary event is called a **Bernoulli trial**. It has exactly one parameter: the probability of "success," $p$.

$$X \sim \text{Bernoulli}(p) \quad \Rightarrow \quad X = \begin{cases} 1 & \text{with probability } p \\ 0 & \text{with probability } 1-p \end{cases}$$

**Mean:** $E[X] = p$
**Variance:** $\text{Var}(X) = p(1-p)$

---

## The CV Connection

Bernoulli trials appear throughout CV wherever a binary decision is made:

### 1. Thresholding (Otsu's method, binarization)
Each pixel either exceeds threshold $T$ or it doesn't. The probability $p$ that a pixel exceeds $T$ depends on the image intensity distribution — Otsu's method finds the $T$ that best separates two Bernoulli populations.

### 2. Feature matching (SIFT, ORB, BRIEF)
Each descriptor pair is either a true match or not. The probability $p$ that a pair is a true match depends on the descriptor distance threshold.

### 3. Sensor physics (the underlying hardware)
A CMOS photosite converts photons to electrons via the **photoelectric effect**. The **quantum efficiency** (QE) of the sensor is the probability that a single photon produces a detectable electron:

$$\text{QE} = p = \frac{\text{electrons produced}}{\text{photons incident}}$$

Modern sensors have QE ≈ 0.4–0.9 (40–90%). Each photon hitting the sensor is an independent Bernoulli trial with success probability $p = \text{QE}$.

---

## Key Insight

Each "1" in the outcome array is one Bernoulli success — one photon absorbed, one pixel above threshold, one descriptor match. The total count of successes is what we care about.

**But wait** — we just counted the total successes out of $n$ trials. That's exactly the **Binomial distribution**.
