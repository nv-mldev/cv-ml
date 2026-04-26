# Stochastic Processes — Time-Indexed Randomness

> **Status: placeholder.** This page will become the dedicated treatment of stochastic processes, covering everything that random *experiments* (the subject of [`part0_what_is_a_distribution.md`](part0_what_is_a_distribution.md)) deliberately leaves out: time indexing, autocorrelation, stationarity, power spectral density, and ergodicity.
>
> The main probability series (parts 1–6 + applied_sensors) only needs single random variables, so this material is not on the critical path for those parts. Read it after part 0 once the question *"what about a whole window of samples, not just one?"* starts to matter — typically when you reach Part II (Signals and Measurement) or any time-series work.

---

## Why This Page Exists

Part 0 introduces the **random process** (single trial, no time index) and notes in passing that a **stochastic process** is the time-indexed version. That distinction is enough to keep parts 1–6 honest, but it leaves real questions on the table:

- What does it mean for two samples in a vibration stream to be "related"?
- Why does a 1 s window of accelerometer data look noticeably different from a 10 s window of the same machine?
- How do you describe a noise signal whose statistics drift over an exposure?
- What's actually being measured by a power spectral density plot?
- When does averaging *over time* tell you the same thing as averaging *over many trials*?

Each of these needs the language of stochastic processes. This page builds that language from first principles.

---

## Intended Outline

The sections below are placeholders. Each will follow the same applied-first pattern as the rest of the probability series: **one concrete sensor scenario → the math needed to describe it → numbers → cross-domain examples.**

### 1. From Random Variable to Stochastic Process

- A stochastic process as **a collection of random variables indexed by time**: $\{X(t) : t \in T\}$
- The two ways to look at one: **a single realisation** (one waveform) vs **the ensemble** (the distribution at each $t$)
- Discrete-time vs continuous-time processes
- Worked example: vibration sensor stream as a discrete-time stochastic process

### 2. Mean and Variance Over Time

- $\mu(t) = \mathbb{E}[X(t)]$ — the mean function
- $\sigma^2(t) = \text{Var}(X(t))$ — the variance function
- Why these can change over time — and why we often *want* them not to
- Worked example: motor warm-up — the noise floor's $\sigma$ changes over the first 10 minutes

### 3. Autocorrelation — Are Successive Samples Independent?

- Definition: $R_X(t_1, t_2) = \mathbb{E}[X(t_1) X(t_2)]$
- Intuition: how much knowing $X(t_1)$ tells you about $X(t_2)$
- The autocorrelation function $R_X(\tau)$ for a stationary process
- Worked example: white noise vs band-limited noise vs a periodic signal — three very different autocorrelation shapes
- Why this matters: many statistical tools (CLT included) assume independent samples; autocorrelation tells you when that assumption is broken

### 4. Stationarity

- **Strict-sense stationary (SSS)**: the joint distribution is invariant to time shifts
- **Wide-sense stationary (WSS)**: only $\mu$ and $R_X(\tau)$ are time-invariant
- Why WSS is what engineers actually use
- How to *check* for stationarity in a real signal (rolling-mean / rolling-variance plots)
- Worked example: a healthy motor as WSS; a degrading one as non-stationary

### 5. Power Spectral Density (PSD)

- The Wiener–Khinchin theorem: $S_X(f) = \mathcal{F}\{R_X(\tau)\}$
- Reading a PSD plot: where is the energy?
- Why PSD is the natural language for vibration, audio, and EEG analysis
- Worked example: bearing fault frequency as a peak in the PSD that wasn't there on the healthy machine

### 6. Ergodicity

- The question: when does a **time average** equal an **ensemble average**?
- $\frac{1}{T}\int_0^T X(t)\,dt \;\overset{?}{=}\; \mathbb{E}[X]$
- Why ergodicity matters: it's the assumption that lets you estimate population statistics from one long recording
- Worked example: estimating noise-floor $\sigma$ from a 60 s vibration recording instead of repeating the experiment 1,000 times

### 7. Common Stochastic Processes

A short tour of the named processes that show up everywhere in signal processing and ML:

- **White noise** — the canonical "memoryless" process; flat PSD
- **Coloured noise** — pink, brown, etc.; PSD shapes you'll see in real sensors
- **Random walk** — cumulative sum of white noise; non-stationary, integrated noise
- **Wiener process (Brownian motion)** — continuous-time random walk; appears in finance, diffusion models, SDE-based generative models
- **Markov chains** — discrete-state, memoryless transitions; foundational for HMMs and reinforcement learning
- **Poisson process** — continuous-time generalisation of the Poisson distribution from [`part3_poisson.md`](part3_poisson.md); arrival times of rare events

### 8. Where Stochastic Processes Show Up Downstream

| Area | What stochastic processes give you |
|---|---|
| Vibration / condition monitoring | PSD-based fault detection; stationarity checks for "is the machine state changing?" |
| Audio | PSD, spectrograms, noise modelling for speech enhancement |
| Image sensors | Fixed-pattern noise vs read noise vs shot noise — different temporal correlation structure |
| Time series forecasting | Stationarity is a prerequisite for ARIMA; autocorrelation drives model order |
| Reinforcement learning | Markov decision processes, return distributions |
| Diffusion generative models | The forward process is a Wiener process; the reverse process is a learned SDE |
| Kalman filtering | Linear-Gaussian stochastic processes are the entire substrate |

---

## Until This Page Is Filled In

The vocabulary table below is the minimum useful summary. It's the same one in [`part0_what_is_a_distribution.md`](part0_what_is_a_distribution.md) §6 — repeated here so this file is self-contained.

| Term | What it means | Time involved? |
|---|---|---|
| **Random process / experiment** | Any procedure with an uncertain outcome | No |
| **Stochastic process** | A collection of random variables indexed by time: $\{X(t) : t \in T\}$ | Yes |
| **Time series** | Observed data from a stochastic process | Yes |
