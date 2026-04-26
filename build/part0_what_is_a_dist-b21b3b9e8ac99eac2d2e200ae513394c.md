# What Is a Probability Distribution?

## Learning Objective

Build the complete conceptual ladder — from a **random process** all the way to a **probability distribution** — using two running examples: a synthetic word source (clean, finite, easy to count) and a vibration sensor stream (noisy, physical, the real thing). By the end of this part you will understand not just *what* a distribution is, but *why* it exists and how every distribution in the rest of this series sits on this same scaffolding.

---

## 1. The Random Process

A **random process** (also called a random experiment) is any procedure that:

1. Has a well-defined set of possible outcomes
2. Produces exactly one outcome each time it runs
3. You cannot predict *which* outcome before it runs

**Example A — the word source:**
Imagine a source that emits a random 3-letter word every time you press a button. Each press is one **trial**. The outcome is one word — say `"cat"`, `"zzq"`, or `"the"`. You can't predict the next word; you only know the rules of the source.

**Example B — the vibration sensor:**
A motor has an accelerometer bolted to it. Every millisecond, the sensor reports a single number — the vibration amplitude in g. Each report is one trial. The outcome is one floating-point value. You can't predict the next reading exactly; you only know the physics of the system and the statistics of its noise.

The two examples differ in almost every surface detail (discrete vs continuous, synthetic vs physical, finite vs unbounded), but the **structure is identical**: a procedure, an outcome, irreducible uncertainty.

The key point: **the randomness is in the process, not in the math.** The math just describes it.

---

## 2. The Sample Space

The **sample space** $\Omega$ is the complete set of all possible outcomes.

For the word source:

$$\Omega = \{\text{all 3-letter combinations}\} = \{aaa, aab, \ldots, zzz\}, \quad |\Omega| = 26^3 = 17{,}576$$

For one accelerometer sample (assuming a 16-bit ADC with $\pm 16$ g range):

$$\Omega = \{-16, -16 + \Delta, \ldots, +16\} \text{ g}, \quad |\Omega| = 2^{16} = 65{,}536 \text{ discrete levels}$$

For a Bernoulli trial (does this sample exceed the alarm threshold?):

$$\Omega = \{0, 1\}$$

The sample space defines the **universe** of what can happen. Every subsequent concept lives inside it.

---

## 3. Events

An **event** is a *subset* of the sample space — a question you ask about the outcome.

| Question | Event (subset of $\Omega$) |
|---|---|
| "Does the word contain a vowel?" | {all words with $\geq 1$ vowel} |
| "Does the word start with 'c'?" | {cat, cab, cod, …} |
| "Did this sample exceed +2 g?" | {samples above +2 g} |
| "Is this sample within the noise floor?" | {samples in $[-0.005g, +0.005g]$} |

The **probability of an event** $P(A)$ is the total probability of all outcomes in that subset.

For a uniform word source (all words equally likely):

$$P(\text{"word starts with 'c'"}) = \frac{26^2}{26^3} = \frac{1}{26} \approx 0.038$$

Events let you ask *questions* about a process. But to do real math, you need numbers — and that's where the random variable comes in.

---

## 4. The Random Variable

A **random variable** $X$ is a function that maps each outcome $\omega \in \Omega$ to a **single real number**.

$$X : \Omega \rightarrow \mathbb{R}$$

It does not change the process. It just attaches a number to each outcome so you can compute with it.

**The word source — three different random variables, same process:**

| Random variable $X$ | Rule | Possible values |
|---|---|---|
| Number of vowels | count(a, e, i, o, u) in word | {0, 1, 2, 3} |
| Alphabetic position of first letter | a→1, b→2, …, z→26 | {1, 2, …, 26} |
| Is the word a real English word? | 1 if yes, 0 if no | {0, 1} |

Same button press, same word — three different numbers depending on the question you ask.

**The vibration sensor — three different random variables, same process:**

| Random variable $X$ | Rule | Possible values |
|---|---|---|
| Raw amplitude (g) | the sensor reading itself | any value in $[-16, +16]$ |
| Did it cross +2 g? | $1$ if yes, $0$ if no | $\{0, 1\}$ |
| Magnitude $\lvert X \rvert$ (g) | absolute value of the reading | any value in $[0, 16]$ |

The random variable is the **bridge** between the physical experiment and the mathematics.

---

## 5. Random Variable vs Random Vector

> **A random variable must always map to a single scalar** — one number on the real line $\mathbb{R}$. This is the strict mathematical definition.

But many real outcomes are naturally multi-dimensional. A 1-second vibration window is 1,000 samples (a 1,000-d vector). A spectrogram column is hundreds of frequency bins. An RGB image patch is thousands of pixel values. Probability theory handles this with a more general object:

| Object | Maps to | Example |
|---|---|---|
| **Random variable** | $\mathbb{R}$ (one scalar) | One sensor reading, one word's vowel count |
| **Random vector** | $\mathbb{R}^n$ ($n$ scalars) | A 1 s window of 1,000 samples; an FFT bin vector |
| **Random matrix** | $\mathbb{R}^{m \times n}$ | A spectrogram; a covariance matrix; an image |

Almost every real downstream model — filtering, classification, neural networks — operates on random vectors and matrices, not single scalars. But the language we develop in this series (Bernoulli, Binomial, Poisson, Normal) lives in $\mathbb{R}^1$ on purpose — you have to understand one-dimensional distributions before high-dimensional ones make sense.

> **Going further:** the deep end of this — high-dimensional distributions, manifolds, what neural networks actually learn, generative models, adversarial examples — is collected in the [companion aside](aside_high_dim_distributions.md). Skip it for now if you're following the main thread; come back when you've finished parts 1–6 and met random vectors in earnest.

---

## 6. Random Process vs Stochastic Process — A Common Confusion

Now that random variables and random vectors are in hand, the next vocabulary trap is easier to clear up.

The word "process" in everyday English implies something happening over time. In probability theory it does not.

| Term | What it means | Time involved? |
|---|---|---|
| **Random process / experiment** | Any procedure with an uncertain outcome | No |
| **Stochastic process** | A collection of random variables indexed by time: $\{X(t) : t \in T\}$ | Yes |
| **Time series** | Observed data from a stochastic process | Yes |

A random experiment is a single trial — one button press, one sensor sample, one coin flip. There is no time axis. You could repeat it, but the repetitions are independent.

A stochastic process is specifically about how a random variable **evolves over time** — the vibration amplitude across an entire one-second window, sensor temperature drift over an hour, network packet rates throughout the day. With the random-vector vocabulary from §5, you can state it precisely: a stochastic process *is* a (possibly infinite) random vector whose components are indexed by time.

| Situation | Which one? |
|---|---|
| One vibration sample at one millisecond | Random experiment — one trial |
| One word from the word source | Random experiment — one trial |
| Vibration amplitude across a 1 s window | Stochastic process — $X(t)$ indexed by time $t$ |
| Sensor temperature drift over 24 h | Stochastic process — how the underlying parameters move |
| Background noise floor changing with thermal load | Stochastic process — how $P(\text{reading})$ shifts over time |

Throughout the rest of this series (parts 1–6), **random process means random experiment** — a single trial with an uncertain outcome. When time indexing matters, we'll say **stochastic process** explicitly.

> **Going further:** stationarity, autocorrelation, power spectral density, ergodicity, and the named processes (white noise, Wiener, Markov, Poisson) all live in the dedicated [`stochastic_processes.md`](stochastic_processes.md) page. That page is currently a placeholder outline — fill-in will come once Part II (Signals and Measurement) needs it.

---

## 7. Why Mean Alone Is Not Enough

Once you have a distribution, what can you do with it that you cannot do with the mean?

Consider two 1-second vibration windows from the same machine — both with mean amplitude 0 g. Are they telling the same story?

No. One window could be a quiet noise floor (small spread, no excursions). The other could be a rhythmic impact pattern (small mean by symmetry, but huge spread and obvious peaks). The mean is a single-point summary. The distribution is the **complete description**.

| What you know | What it tells you | What it misses |
|---|---|---|
| Mean only | The center | Spread, shape, tails, symmetry |
| Mean + std | Center + typical spread | Shape, skewness, tail behaviour |
| Full distribution | Everything | Nothing — it's the complete model |

---

## 8. The Idea vs the Reality — Distribution vs Histogram

**Probability Distribution (the blueprint)**
- Theoretical — a model we assume or derive from first principles
- Smooth curve (PDF) or exact probabilities (PMF)

**Data Distribution / Histogram (what you measure)**
- Finite, noisy — what you actually observe from an experiment
- Converges to the true distribution as you collect more data

> The histogram is your approximation. The distribution is what it's approximating.

Knowing how many samples you need before the histogram is a reliable estimate of the distribution is one of the core engineering skills in any signal-processing or measurement system.

---

## 9. Reading a Distribution: Shape → Physical Meaning

| Shape feature | Mathematical name | Physical meaning |
|---|---|---|
| **Center** | Mean ($\mu$) or mode | Expected value of the measurement |
| **Width** | Standard deviation ($\sigma$) | Noise level — how much the measurement spreads |
| **Asymmetry** | Skewness | Clipping at sensor limits squashes the tail |
| **Heavy tails** | Kurtosis | Rare events — shock pulses, hot pixels, outlier readings |
| **Multiple peaks** | Multimodality | Mixture of regimes — e.g., bearing alternating between healthy and faulty states |

---

## 10. Parameters Compress the Distribution

A distribution can be fully described by a small number of **parameters**.

For the vowel-count example, two numbers fully determine the shape:

$$n = 3 \quad (\text{letters per word}), \qquad p = 5/26 \quad (\text{vowel probability})$$

| Distribution | Parameters | What they control | Info needed |
|---|---|---|---|
| Bernoulli | $p$ | success probability | 1 number |
| Binomial | $n$, $p$ | number of trials, success prob | 2 numbers |
| Poisson | $\lambda$ | expected count | 1 number |
| Normal | $\mu$, $\sigma$ | center, spread | 2 numbers |

Parameters are the **compression algorithm for uncertainty**. A Gaussian noise model needs only $\mu$ and $\sigma$ to describe the entire shape of a sensor's noise floor — that's the entire point of having a distribution rather than just storing every measurement.

(Once we get to high-dimensional distributions over images, audio, or video, the parameter count balloons from 2 to billions — but that's a story for the [companion aside](aside_high_dim_distributions.md), not this part.)

---

## 11. The Complete Ladder

```
Random Process           — press the button / capture a sample / expose a sensor
    ↓
Sample Space Ω           — all 17,576 words / all 65,536 ADC levels / {0,1}
    ↓
Event A ⊆ Ω              — "word has ≥1 vowel" / "sample exceeds +2g"
    ↓
Random Variable X        — vowel count (scalar) / sample magnitude (scalar)
Random Vector x          — 1 s window of 1,000 samples / FFT bin vector
    ↓
Distribution             — P(X=k) table / Poisson(λ) / Normal(μ,σ)
    ↓
Parameters               — (n, p) / λ / (μ, σ)
```

Every probability statement we make in parts 1–6 lives somewhere on this ladder. The right branch (high-dimensional distributions over images, audio, learned latent spaces) is built on the same scaffolding — see the [companion aside](aside_high_dim_distributions.md) when you're ready.

---

## 12. Why Different Situations Need Different Distributions

| Process | What's random | Key constraint | Distribution |
|---|---|---|---|
| Sample above threshold or not | Binary outcome | Only 0 or 1 | **Bernoulli** |
| How many samples in a window exceed threshold | Count of successes | Bounded by $n$ | **Binomial** |
| Rare events (shock pulses, packet drops, photons) | Rare events, large $n$, small $p$ | Mean = variance | **Poisson** |
| Filter output, averaged window, sensor noise floor | Sum of many terms | Any real value | **Normal** |
| High-dimensional signal (image, audio clip, video) | Many random variables jointly | Lives on a manifold | **Deep generative model** (see aside) |

**In the next parts** we derive each distribution from first principles — Bernoulli → Binomial → Poisson → Normal — building the mathematical foundation for everything above.
