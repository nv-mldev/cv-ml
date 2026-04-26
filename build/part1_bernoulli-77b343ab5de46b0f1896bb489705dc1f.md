# Part 1: The Bernoulli Trial — The Atom of Randomness

## An Applied Scenario — One Reading from a Vibration Sensor

A factory motor has an accelerometer bolted to its housing. Every millisecond, the sensor reports a vibration amplitude in g (units of gravity). Most of the time the reading sits inside the normal operating band — say, between -2 g and +2 g. Occasionally, a transient impact pushes it above +2 g.

You set up a simple condition monitor: **for each new sample, mark it `1` if it exceeds +2 g, else `0`.**

That single observation — *did this sample cross the threshold?* — is the smallest possible random event. There are exactly two outcomes. There is one number that controls how often you see a `1`: the probability that any given sample is a transient.

You now have everything you need to define the simplest distribution in probability.

---

## Intuition

The atom of all the randomness we'll study is this: **something either happens or it doesn't.**

- A vibration sample either crosses the alarm threshold or it doesn't (condition monitoring)
- A motor bearing either fails on a given day or it doesn't (reliability)
- A photon either gets absorbed by a sensor pixel or it doesn't (image sensor physics)
- A coin lands heads or tails

This binary event is called a **Bernoulli trial**. It has exactly one parameter: the probability of "success," $p$. ("Success" just means the event you're counting — a threshold crossing, a fault, an absorbed photon.)

$$X \sim \text{Bernoulli}(p) \quad \Rightarrow \quad X = \begin{cases} 1 & \text{with probability } p \\ 0 & \text{with probability } 1-p \end{cases}$$

**Mean:** $E[X] = p$
**Variance:** $\text{Var}(X) = p(1-p)$

The mean tells you the long-run fraction of `1`s. The variance is largest at $p = 0.5$ (maximum uncertainty) and shrinks to $0$ as $p$ approaches $0$ or $1$ (the outcome becomes predictable).

---

## Back to the Vibration Sensor

For the motor scenario, suppose calibration runs show that on a healthy machine, roughly 1 sample in 200 crosses +2 g from ambient noise alone. That fixes the parameter:

$$p = \frac{1}{200} = 0.005$$

Each new millisecond is one Bernoulli trial with $p = 0.005$. The stream of `0`s and `1`s coming out of the threshold check is a sequence of independent Bernoulli outcomes — provided the machine state isn't changing.

This is already useful on its own: if you start seeing `1`s at a rate much higher than $0.005$, something has changed. But to *quantify* "much higher" you need to count successes over a window of $n$ samples — and that's the Binomial distribution.

---

## Where Else Bernoulli Trials Appear

The same atom shows up across signal processing and ML wherever a single binary decision is made:

| Domain | What's the trial? | What's $p$? |
|---|---|---|
| Vibration monitoring | One sample crosses alarm threshold | Probability of a transient per sample |
| Audio VAD | One frame contains speech | Speech-active fraction |
| Image sensor (CMOS) | One photon produces a detectable electron | Quantum efficiency, QE |
| Image thresholding | One pixel exceeds intensity $T$ | Fraction of "bright" pixels |
| Binary classifier | One input is classified as positive | Class prior × model accuracy |
| Reliability | One unit fails in its first year | Annual failure probability |

**A note on the photon case** — it's the cleanest physical Bernoulli trial in nature. A CMOS photosite converts photons to electrons via the photoelectric effect. The sensor's **quantum efficiency** (QE ≈ 0.4–0.9 for modern silicon) is *literally* a Bernoulli probability:

$$\text{QE} = p = \frac{\text{electrons produced}}{\text{photons incident}}$$

Each incident photon is one independent trial. We'll lean on this in later parts because the physics gives us an exact $p$ to work with.

---

## Key Insight

Each `1` in a Bernoulli stream is one event — one threshold crossing, one absorbed photon, one positive classification. On its own, a single trial tells you almost nothing. What you actually care about is the **count of successes over many trials**: how many transients in a 1-second window, how many electrons from a 10 ms exposure, how many positive predictions in a batch.

Counting Bernoulli successes is exactly what the **Binomial distribution** does — and that's Part 2.
