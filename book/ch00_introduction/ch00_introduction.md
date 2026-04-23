# Chapter 0 — From Measurements to Meaning

## Why This Book Exists

This book is about one problem, stated many ways:

- How do we recover the true value of something when our measurements of
  it are noisy?
- How do we extract *meaning* from raw sensor readings?
- How do we turn a grid of pixels into a decision: *what is this?*
- How do we train a model that generalizes from examples it has seen to
  data it hasn't?

These are all the same problem. It is called the **inverse problem** — given
the effects (measurements), recover the causes (the underlying truth). It
is arguably the central problem of empirical science and engineering.

The three major approaches to attacking it — classical signal processing,
parametric model fitting, and flexible machine learning — define the
structure of this book. By the end, you will know when to reach for each,
and why modern computer vision and multimodal AI lean so hard on the
third.

---

## 1. The Clean World

Forget models for a moment. You are an engineer with a sensor.

- You have a **source** — some thing in the world with a value you want to
  measure. A temperature. A distance. A concentration. A pixel intensity
  at a particular location on a scene.
- You point a **sensor** at it and read out a number — the **measurement**.
- If the sensor were perfect, the measurement would equal the true value
  exactly.

Formally: we have an input $x$ (what we control, or what identifies the
source) and a measurement $y$ we read off the sensor. There is some
relationship

$$
y = f(x)
$$

where $f$ is whatever the underlying physics dictates — linear,
exponential, wavy, arbitrary.

**In a perfect world, machine learning wouldn't exist.** If you knew $f$,
you'd just plug in $x$ and read off $y$. No learning, no estimation, no
uncertainty. The entire ML and signal-processing industry hinges on
everything that comes next.

![The clean world: a perfect sensor reading out the exact value of $f(x)$.](ch00_clean_world.png)

---

## 2. Noise Enters

Real sensors don't give you $f(x)$. They give you
$f(x) + \text{garbage}$. The garbage has physical origins:

- **Thermal noise** — electrons jiggling because the sensor is at nonzero
  temperature.
- **Quantization** — the ADC rounds to the nearest integer count.
- **Calibration drift** — the sensor baseline changes over hours or days.
- **Stray light / EM interference** — unintended signals leaking in.
- **Mechanical vibration** — the source or the sensor moved slightly
  between readings.

You cannot predict the garbage on any particular reading. What you *can*
predict is its **statistics** — its distribution, its mean (often 0), its
variance.

### Running the same measurement many times

Hold $x$ fixed. Read the sensor 1000 times. In the clean world the
readings would all be identical. In reality they scatter:

![1000 repeated readings at a fixed x. Left: the sequence of values. Right: their distribution.](ch00_repeated_readings.png)

Two observations that matter for everything that follows:

1. **Single readings are almost useless.** No individual reading equals
   the truth. The best we can say is *"probably within a couple of
   $\sigma$ of the truth."*
2. **The readings aren't random in a lawless way.** They cluster — the
   distribution has a **shape**, a mean, a spread, symmetry. Noise is
   unpredictable at the level of one sample but predictable at the
   level of many.

That second point is the foundation of everything. We can't beat noise on
a single reading, but we can *characterize* it well enough to design
algorithms that work on average. The mathematical name for this is
**statistics**.

---

## 3. Why Is the Noise Gaussian?

Look at the histogram in Section 2. That bell shape isn't coincidence. In
physics and engineering, measurement noise is **overwhelmingly Gaussian**,
and there is a deep reason: the **Central Limit Theorem (CLT)**.

The CLT says: if you add up many independent small random contributions,
each from some distribution (any distribution, as long as each has a
finite variance), the **sum** tends to be Gaussian-distributed —
regardless of the individual distributions.

Your sensor's noise is the sum of many tiny independent contributions:
thermal, quantization, vibration, EM. By the CLT, their aggregate is
approximately Gaussian. **This is physics, not a mathematical
convenience.**

![CLT in action: summing more and more uniform (non-Gaussian) samples produces a bell curve.](ch00_clt.png)

> **Part I detour:** the full probability toolkit that makes this
> statement precise — Bernoulli → Binomial → Poisson → Normal → CLT — is
> built in [Part I of this book](../../math/probability/README.md). If
> your probability is rusty or if you want the derivations, read that
> first. If you trust the intuition above for now, continue.

The practical payoff arrives in Chapter 7 (Part IV), where we build on
this intuition to derive least-squares fitting as maximum likelihood
under Gaussian noise — a rigorous justification for why so many
algorithms in CV and ML use squared-error losses.

---

## 4. The Inverse Problem — Three Attacks

Now we can state the general problem clearly.

> **Given** noisy measurements $y_i = f(x_i) + \epsilon_i$, where
> $\epsilon_i$ is random with approximately known statistics.
> **Goal:** recover something useful about $f$ — specific values, the full
> function, or predictions at new inputs.

Three attacks exist. Each makes a different assumption about how much you
already know about $f$ before you start.

### Attack 1 — Averaging and signal processing

**Premise:** I can repeat the measurement at the same $x$ as many times
as I want.

Take $N$ readings at a single $x$. Their average $\bar{y}$ has expected
value $f(x)$ (the noise averages out) and standard deviation
$\sigma / \sqrt{N}$. Double the readings → noise drops by $\sqrt{2}$.
This is the famous **$\sqrt{N}$ rule** — it is the whole reason that
scientific instruments have "integration time" knobs.

Classical signal processing generalizes averaging — low-pass filtering,
Wiener filtering, Kalman filtering — all are sophisticated forms of
"combine many noisy observations to reduce uncertainty."

- **Buys you:** excellent estimates at the specific $x$ values you
  measured.
- **Doesn't buy you:** any predictions for *new* $x$ values.

Parts II and III of this book develop this attack for the imaging case:
sampling, sensors, pixels, contrast, and why raw-pixel operations run
into fundamental limitations.

### Attack 2 — Parametric fitting (known model form)

**Premise:** I already know the functional form of $f$ from physics or
from prior knowledge. I just don't know a handful of constants.

Examples:

- Radioactive decay: $y(t) = A e^{-\lambda t}$ — two unknowns $A, \lambda$.
- Sensor calibration: $y = \alpha x + \beta$ — slope and offset.
- Ideal gas: $PV = nRT$ — fit $n$ to data.

Pick the constants that make the model best match the data (usually
least-squares). This is what classical statistics calls **regression**.
You're not discovering what $f$ looks like; you're nailing down a few
numbers inside a form that was handed to you by domain knowledge.

- **Buys you:** predictions at any $x$, not just the measured ones.
- **Costs:** if the assumed form is wrong, your predictions are wrong no
  matter how much data you collect.

### Attack 3 — Flexible learning (machine learning)

**Premise:** I don't know the form of $f$. But I have many $(x, y)$ pairs
and I'm willing to spend compute.

Pick a flexible **hypothesis class** — polynomials, kernels, neural
networks, transformers — and find the member that best matches the data.
You're not committing to a specific form, just a space of forms. The
algorithm chooses the form from the space.

- **Buys you:** the ability to handle problems where no physical model
  exists — image classification, language, complex real-world mappings.
- **Costs:** much more data, careful handling of overfitting, harder
  interpretation of the resulting model.

Parts V (CNNs) and VI (attention, vision transformers, multimodal models)
of this book develop Attack 3. They are, structurally, elaborate
parametric-fitting problems — but with hypothesis classes flexible
enough to learn the form of $f$ rather than inherit it.

---

## 5. Three Attacks on the Same Data

To make the three attacks tangible, simulate a small noisy dataset and
attack it three ways:

![The same noisy dataset under three different attacks: averaging (a sharp point estimate at one x), linear fit (predicts everywhere but misses the wiggle), degree-10 polynomial (tracks the wiggle but risks overfitting).](ch00_three_attacks.png)

What the three attacks tell us:

- **Attack 1 (averaging)** gives a very good value at one specific $x$ —
  the error bar on the averaged estimate is tiny. But we have no idea
  what $f$ does elsewhere.
- **Attack 2 (linear fit)** predicts everywhere but misses the wiggle. If
  we *knew* from physics that $f$ was linear, this would be the right
  tool. Wrong assumption, wrong answer.
- **Attack 3 (polynomial fit)** tracks the wiggle well. With less data or
  a higher degree it would start fitting the noise instead of the
  signal — the failure mode called **overfitting**. Controlling it is
  half of what modern ML is about.

**No attack is universally right.** The skill is picking the attack that
matches what you know about your problem. In practice you often combine
them — e.g. average noisy pixel values first (Attack 1), then fit a
calibration curve to the averages (Attack 2), then use a neural network
on the calibrated data (Attack 3).

---

## 6. From 1D Signals to Images to Multimodal AI

Everything above used a scalar input $x$. Real problems are almost
always higher-dimensional:

- **A pixel in an image** — input is a 2D spatial coordinate, output is
  the intensity at that coordinate.
- **A full image** — input *is* the image (a high-dimensional vector of
  pixels), output is a classification or a latent feature.
- **Video** — input is an image indexed by time; noise and signal both
  have temporal structure.
- **Audio** — input is a 1D signal over time; same framework, different
  dimensionality.
- **Multimodal** — an input might contain an image *and* text *and*
  audio simultaneously; each modality is a different signal, and the
  task is to fuse them.

The mathematics scale cleanly: wherever we wrote $y = f(x) + \epsilon$ for
scalar $x$, we can write $y = f(\mathbf{x}) + \epsilon$ for vector
$\mathbf{x}$, or $\mathbf{y} = f(\mathbf{x}) + \boldsymbol{\epsilon}$ for
vector output. The three attacks stay the same. Visualization gets
harder, the amount of data needed grows (the **curse of dimensionality**),
and the algorithms become heavier — but the problem statement doesn't
change.

This is why the book's title is **Signals to Transformers** and not
*Pixels to Transformers*: the framework subsumes pixels, tokens, audio,
and video alike. A transformer processing a paragraph, a ViT processing
an image, and a CLIP model fusing images with captions are all solving
the same inverse problem — they just work in different signal spaces.

---

## 7. How to Read This Book

You don't have to read linearly. Three paths are supported:

### Path A — Top-to-bottom, math first
If you want the mathematical foundations laid in properly before any
applied content, read **Part I (Math Foundations)** in full, then Parts
II–VI in order.

Good if: you've worked in engineering adjacent to CV but the probability
/ linear algebra is rusty.

### Path B — Applied first, math as needed
Start with **Part II (Signals and Measurement)**, work through the rest
in order, and use Part I as a reference when the math gets too thin.

Good if: you already know undergrad probability and linear algebra and
want to get to CV fast.

### Path C — Target a specific chapter
Every chapter lists its prerequisites up front. Jump to whatever you need
(e.g. Chapter 13 on self-attention if that's why you're here) and
backtrack to prerequisite chapters as needed.

Good if: you have a specific goal in mind and your foundations are already
solid.

### When to read the sibling `nn-basics` notebooks

For any chapter that involves *training* a model from scratch, there is a
matching drafting notebook in
[`~/projects/nn-basics/fundamentals/`](../../../nn-basics/fundamentals/README.md).
Those notebooks are `.ipynb` format for interactive exploration — rerun
cells, tweak hyperparameters, watch the loss curves. Use them as
playgrounds for ideas the book describes statically.

### Reading checkpoints — the four big "aha" moments

If you only get these four, the book has done its job:

1. **Why squared-error is special** — it's the maximum-likelihood estimator
   under Gaussian noise (Chapter 7).
2. **Why linear algebra underlies every vision operation** — image
   comparison, matching, features, and attention all reduce to dot
   products and projections (Chapter 8).
3. **Why convolutions work for images** — weight sharing over a
   translation-invariant domain (Chapter 9).
4. **Why attention works for everything else** — dynamic, data-dependent
   aggregation without baked-in geometry (Chapter 13).

---

## Summary

| Concept | Key idea |
|---------|----------|
| Measurement | $y = f(x) + \epsilon$ — signal plus noise |
| Noise | Physical, statistical, typically Gaussian (CLT) |
| Inverse problem | Recover $f$ from $(x, y)$ pairs |
| Attack 1 | Average / filter at known $x$ values (signal processing) |
| Attack 2 | Fit parameters inside a known functional form (regression) |
| Attack 3 | Pick a flexible hypothesis class, let data choose the form (ML) |
| Signals | Pixels, tokens, audio, video — the same math covers all |

---

**Next →** [Part I — Math Foundations](../../math/probability/README.md)
if you want the probability and linear algebra before the applied content,
or skip to [Part II — Signals and Measurement](../part2_signals_and_measurement/ch01_digitisation/)
for the first applied chapter.
