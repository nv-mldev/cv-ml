# Introduction to Probability for Computer Vision

## Learning Objective

Build the complete conceptual ladder — from a **random process** all the way to a **probability distribution** — using a single running example. By the end of this part you will understand not just *what* a distribution is, but *why* it exists and *how* it connects to every CV algorithm that follows, all the way up to modern generative models.

---

## 1. The Random Process

A **random process** (also called a random experiment) is any procedure that:

1. Has a well-defined set of possible outcomes
2. Produces exactly one outcome each time it runs
3. You cannot predict *which* outcome before it runs

**Example — the word source:**
Imagine a source that generates a random 3-letter word every time you press a button.
Each press is one **trial**. The outcome is one word — say, `"cat"`, `"zzq"`, or `"the"`.

**In CV, a random process might be:**

- **Capturing one image patch from a camera** — the randomness comes from *which patch* you capture. The world has infinite possible scenes, textures, and objects. Each press of the shutter gives you a different 48×48 region. The sample space is all possible patch appearances.

- **Measuring one pixel's brightness under fixed lighting** — here the scene is fixed: same object, same light. The randomness comes purely from **sensor noise** — shot noise, read noise, thermal noise. Press the shutter twice on the same frozen scene and you get slightly different pixel values each time.

- **Testing whether one descriptor pair is a true match** — a descriptor (SIFT, ORB, HOG) is a compact numerical summary of a local patch — for example, a 128-dimensional vector of gradient information. The random process is: *given two keypoints from two images, are they the same physical point in the scene or not?* You compute the Euclidean distance between their descriptors and make a probabilistic decision — match or no match. That binary decision is a Bernoulli trial. The randomness comes from not knowing in advance whether the two keypoints truly correspond.

The key point: **the randomness is in the process, not in the math.** The math just describes it.

---

### Random Process vs Stochastic Process — A Common Confusion

The word "process" in everyday English implies something happening over time. In probability theory it does not.

| Term | What it means | Time involved? |
|------|--------------|----------------|
| **Random process / experiment** | Any procedure with an uncertain outcome | No |
| **Stochastic process** | A collection of random variables indexed by time: $\{X(t) : t \in T\}$ | Yes |
| **Time series** | Observed data from a stochastic process | Yes |

A random experiment is a single trial — press the button once, capture one patch, expose one pixel. There is no time axis. You could repeat it, but the repetitions are independent.

A stochastic process is specifically about how a random variable **evolves over time** — the pixel value at position (x, y) across video frames, sensor temperature drift over an exposure sequence, background scene appearance changing over hours.

**In CV:**

| Situation | Which one? |
|-----------|-----------|
| Capture one image patch | Random experiment — one trial |
| Measure one pixel's brightness | Random experiment — one observation |
| Pixel brightness at (x,y) across 1000 video frames | Stochastic process — X(t) indexed by frame t |
| Background subtraction in surveillance | Stochastic process — modelling how P(pixel) shifts over time |
| Sensor noise measured repeatedly over time | Stochastic process — how noise drifts with temperature |

Throughout this series, **random process means random experiment** — a single trial with an uncertain outcome. When time indexing matters (video, temporal sequences), we will say **stochastic process** explicitly.

---

## 2. The Sample Space

The **sample space** Ω is the complete set of all possible outcomes.

For the word source:

$$\Omega = \{\text{all 3-letter combinations}\} = \{aaa, aab, \ldots, zzz\}$$

$$|\Omega| = 26^3 = 17{,}576 \text{ possible words}$$

For a single pixel:

$$\Omega = \{0, 1, 2, \ldots, 255\}$$

For a Bernoulli trial (is this pixel an edge?):

$$\Omega = \{0, 1\}$$

The sample space defines the **universe** of what can happen. Every subsequent concept lives inside it.

---

## 3. Events

An **event** is a *subset* of the sample space — a question you ask about the outcome.

| Question | Event (subset of Ω) |
|----------|----------------------|
| "Does the word contain a vowel?" | {all words with ≥1 vowel} |
| "Does the word start with 'c'?" | {cat, cab, cod, …} |
| "Is the pixel bright?" | {128, 129, …, 255} |
| "Is this pixel an edge?" | {1} |

The **probability of an event** P(A) is the total probability of all outcomes in that subset.

For a uniform word source (all words equally likely):

$$P(\text{"word starts with 'c'"}) = \frac{26^2}{26^3} = \frac{1}{26} \approx 0.038$$

Events let you ask *questions* about a process. But to do real math, you need numbers — and that's where the random variable comes in.

---

## 4. The Random Variable

A **random variable** X is a function that maps each outcome ω ∈ Ω to a **single real number**.

$$X : \Omega \rightarrow \mathbb{R}$$

It does not change the process. It just attaches a number to each outcome so you can compute with it.

**The word source — three different random variables, same process:**

| Random variable X | Rule | Possible values |
|-------------------|------|-----------------|
| Number of vowels | count(a,e,i,o,u) in word | {0, 1, 2, 3} |
| Alphabetic position of first letter | a→1, b→2, …, z→26 | {1, 2, …, 26} |
| Is the word a real English word? | 1 if yes, 0 if no | {0, 1} |

Same button press. Same word. Three different numbers depending on the question you ask.

**In CV:**

| Process | Outcome ω | Random variable X |
|---------|-----------|-------------------|
| Capture one patch | 48×48 pixel array | Mean brightness of patch |
| Test one descriptor pair | (descriptor_a, descriptor_b) | Euclidean distance between them |
| Observe one photosite | photon/electron interaction | Number of electrons produced |

The random variable is the **bridge** between the physical experiment and the mathematics.

---

## 5. Random Variable vs Random Vector

> **A random variable must always map to a single scalar** — one number on the real line ℝ. This is the strict mathematical definition.

But in CV, outcomes are often naturally multi-dimensional. A SIFT descriptor is 128 numbers. An image patch is 48×48 = 2,304 numbers. This leads to a more general concept:

| Object | Maps to | Example |
|--------|---------|---------|
| **Random variable** | ℝ (one scalar) | Pixel intensity, match distance, vowel count |
| **Random vector** | ℝⁿ (n scalars) | SIFT descriptor (128-d), image patch (2,304-d) |
| **Random matrix** | ℝ^(m×n) | Full image, covariance matrix |

**In the descriptor matching example:**
- The **random vector** is the descriptor itself — 128 numbers
- The **random variable** is the Euclidean distance between two descriptors — collapses 128 numbers to one scalar
- That scalar is what you threshold to decide match / no match

The distance function is exactly the mapping $X : \Omega \rightarrow \mathbb{R}$ — it takes a pair of descriptors (the outcome) and returns one number.

**Why does this matter?**

Almost everything interesting in CV lives in high-dimensional random vectors. A neural network layer output is a random vector. An image is a random vector. Learning a distribution over those vectors is what makes deep learning hard — and powerful.

---

## 6. Distributions Over High-Dimensional Spaces

Consider a full colour image fed into a neural network. Its dimensions are:

$$3 \times 224 \times 224 = 150{,}528 \text{ dimensions}$$

— where **3** is the number of colour channels (R, G, B) and **224×224** is the spatial size of the image.

So a single image is one point in ℝ^**150,528**. The true distribution P(**x**) over this space has several brutal properties:

### It is wildly multimodal

There is one cluster (mode) for cat images, another for car images, another for faces, another for aerial views — thousands of modes for all visual concepts that exist in the world. The distribution is not a smooth bell curve. It is a jagged, complex, high-dimensional landscape.

### It lives on a low-dimensional manifold

Most random points in ℝ^150,528 look like pure static noise — they are not valid images. The set of all "natural images" occupies a tiny, curved **manifold** embedded inside that vast space. All real photographs live near this manifold. Everything off it is meaningless.

### You can never fully sample it

The number of possible distinct images is incomprehensibly large. Even if every atom in the universe stored one image, you would cover an infinitesimal fraction of the manifold. So the training dataset is always a sparse, finite sample from the true distribution.

**The scale of the problem:**

| Scale | Distribution | Model type | How you represent it |
|-------|-------------|------------|----------------------|
| 1 pixel | P(x), x ∈ ℝ¹ | **Statistical** | Histogram, Poisson, Gaussian — 1 or 2 parameters |
| 1 patch mean | P(x), x ∈ ℝ¹ | **Statistical** | Gaussian noise model — just μ and σ, no learning |
| SIFT descriptor | P(**x**), **x** ∈ ℝ^128 | Statistical or shallow ML | GMM, PCA — too complex for a simple histogram |
| Full image | P(**x**), **x** ∈ ℝ^150,528 | **Deep learning** | Neural network — billions of parameters, learned from data |
| Conditional | P(**x** \| "cat") | **Deep learning** | Conditional generative model (GAN, VAE, diffusion) |

> **Note on "statistical model" vs "deep learning model":** A statistical model is a hand-crafted parametric assumption — you choose the distribution family (Gaussian, Poisson) and estimate its parameters from data. No architecture, no backpropagation. A deep learning model *learns* the distribution family itself from data — the architecture is flexible enough to approximate any distribution given enough parameters and data. Both are models of a distribution. The difference is in how expressive and how automated the representation is.

The math is identical at every scale — it is still just a probability distribution. Only the **representation** of that distribution changes.

---

## 6b. The Network Learns in Feature Space, Not Pixel Space

This is a crucial precision that most introductions skip.

When we say "the network learns the distribution over images", we do not mean it learns P(**x**) directly in the 150,528-dimensional pixel space. What it actually learns is a distribution over a **learned, low-dimensional feature space** — a compressed representation that the network itself constructs through its layers.

### What happens layer by layer

```
Input:  raw image x ∈ ℝ^150,528     ← pixel space, high-dimensional, no semantic structure
    ↓  convolutional layers
    z₁ ∈ ℝ^d₁                       ← low-level features: edges, blobs, colour gradients
    ↓  deeper layers
    z₂ ∈ ℝ^d₂                       ← mid-level: textures, object parts
    ↓  deeper still
    z  ∈ ℝ^d                        ← final feature vector / latent code   (d << 150,528)
    ↓
    output: class label / reconstruction / generated image
```

A ResNet-50 bottleneck is **2,048-d**. A VAE latent space is typically **128–512-d**. Stable Diffusion operates in a **64×64×4 = 16,384-d** latent space — not in the 512×512×3 = 786,432-d pixel space. The network compresses the input dramatically, and the distribution it learns lives in that compressed space.

### Why pixel space is the wrong place to model the distribution

Pixel space has no useful geometric structure for learning:

- Two images that are **semantically identical** (same cat, slightly different lighting) can be thousands of units apart in pixel space
- Two images that are **semantically opposite** (a cat vs a dog) can be pixel-close if they share similar colours and background

**"The manifold exists in pixel space, but similar images are far apart" — what does that mean?**

Think of two photos of the same cat, one taken a millisecond apart. They look almost identical to you. But in pixel space, even a tiny change in lighting shifts *every* pixel value slightly. The Euclidean distance between the two 150,528-dimensional vectors can be large, even though the semantic content is nearly the same.

Conversely, a photo of a grey cat on a grey sofa and a photo of a grey dog on a grey sofa can have *very similar* pixel values (same colours, same background layout), even though the images are semantically opposite.

So the manifold — the curved surface on which all valid cat images sit — mathematically exists inside ℝ^150,528. But that surface has no clean shape in pixel coordinates: it is twisted, folded, and its distances do not match human semantic similarity. Fitting a probability distribution to a shape like that is intractable.

The manifold of natural images exists in pixel space, but it has no clean parametric form there. Fitting a distribution directly in pixel coordinates is intractable.

### Feature space fixes this

The network learns an encoder mapping:

$$f : \mathbb{R}^{150528} \rightarrow \mathbb{R}^d$$

such that:
- Semantically **similar** images map to **nearby points** in ℝ^d
- Semantically **different** images map to **distant points**
- The distribution P(**z**) over the feature space is far simpler and more structured than P(**x**) over pixel space

This is why every major generative architecture works in feature/latent space:

| Architecture | Where the distribution lives | What the distribution looks like |
|-------------|------------------------------|----------------------------------|
| **VAE** | Latent space ℝ^d | Gaussian — P(**z**) = N(0, I) |
| **GAN** | Latent space ℝ^d | Simple prior (Gaussian or Uniform) mapped to images |
| **Stable Diffusion** | Latent space 64×64×4 | Gaussian noise progressively denoised |
| **Contrastive (SimCLR)** | Feature space ℝ^d | Positives clustered, negatives separated |

### The geometric picture

```
Pixel space ℝ^150,528
┌──────────────────────────────────────┐
│  · · · noise · · · · · · · · · · ·  │
│  · · [cat manifold] · · · · · · ·  │  ← thin curved surface in a vast space
│  · · · · · [dog manifold] · · · ·  │
│  · · · · · · · · · [car] · · · ·   │
└──────────────────────────────────────┘
              ↓  encoder  f(x)
Feature space ℝ^d    (d << 150,528)
┌─────────────────┐
│  [cats]         │  ← compact, well-separated clusters
│       [dogs]    │  ← distribution is now tractable
│  [cars]         │
└─────────────────┘
```

The encoder collapses the high-dimensional manifold into a low-dimensional space where the distribution is simple enough to model and sample from.

### Connection back to the distributions in this series

Everything in Parts 1–6 — Bernoulli, Binomial, Poisson, Normal — lives in ℝ¹. One random variable, one number.

The step to deep learning is:
1. Stack many random variables into a random vector → ℝ^d
2. Learn an encoder that maps pixel space to a feature space where the distribution is simple
3. Fit a tractable distribution in that feature space — often approximately Gaussian

This is why the Normal distribution from Part 4 is not just a historical curiosity. It is the **literal prior** used in VAEs:

$$P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The whole series is building the mathematical foundation for that one line.

### Further Reading — Manifold Visualizations

The manifold concept is best understood visually. These are the best freely available resources:

1. **Colah's blog — "Neural Networks, Manifolds, and Topology"** (2014)
   The clearest visual explanation of how neural networks learn to separate data on a manifold.
   Diagrams are CC BY-SA 3.0.
   → https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/

2. **Scikit-learn — Swiss Roll dataset**
   A canonical 3D visualization of a 2D manifold (a curved surface) embedded in 3D space.
   Shows exactly how data concentrates on a thin surface inside a larger ambient space.
   BSD-3 licensed.
   → https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html

3. **Goodfellow, Bengio, Courville — Deep Learning book, Figure 5.11**
   The canonical academic figure showing the manifold hypothesis: natural images occupy
   a tiny curved subspace of the full pixel space.
   Freely readable online.
   → https://www.deeplearningbook.org/contents/ml.html  (scroll to Figure 5.11)

The code in `part0_what_is_a_distribution.py` generates a local version of the Swiss roll manifold visualization with CV-specific annotations — no internet connection required.

---

## 7. What Neural Networks Actually Learn — Interpolation and Extrapolation

Since you can never sample the full distribution, the network must **generalise** — make predictions about unseen data. It does this in two fundamentally different ways:

### Interpolation (the normal case)

A new input falls *between* training samples on the manifold. The network smoothly fills in — it has seen nearby examples and can reasonably estimate the distribution in that region.

> Example: you train on thousands of cat images. A new cat image with slightly different lighting falls between training examples on the manifold. The network interpolates and gets it right.

This is why neural networks generalise to unseen images that are **similar** to training data.

### Extrapolation (the dangerous case)

A new input falls *outside* the region covered by training samples — off the manifold, or in a region with no nearby training examples. The network is guessing, and it can fail badly.

> Example: a medical imaging model trained on Western hospital data sees an image from a different scanner or demographic. It extrapolates into unknown territory and its predictions become unreliable.

This is why models:
- **Hallucinate** — generate confident but wrong outputs
- **Fail on distribution shift** — perform well in training but fail in deployment
- **Are vulnerable to adversarial examples** — tiny perturbations push inputs off the manifold into regions where the network extrapolates wildly

---

### Distribution Shift — The World Moves, Not the Model

Distribution shift happens when the training distribution P_train(**x**) and the deployment distribution P_deploy(**x**) are different. The network learned a good approximation of P_train — but at deployment it receives inputs from P_deploy, a region it has never seen.

The model is not wrong. It learned exactly what it was shown. The *world moved*.

| Trained on | Deployed on | What shifts |
|-----------|-------------|-------------|
| Daytime road images | Night-time driving | Lighting distribution |
| Western hospital CT scans | Asian hospital CT scans | Scanner type + patient demographics |
| Clean studio photos | Blurry phone camera photos | Image quality distribution |
| Summer dataset | Winter dataset | Colour, texture, scene distribution |

In every case the model is extrapolating into a region of ℝ^150,528 it has never covered. It may still produce a confident-looking output — but that confidence is not trustworthy. This is one of the hardest real-world problems in CV deployment.

---

### Adversarial Examples — Not the Same as Negative Examples

Before explaining adversarial examples, it is important to distinguish them from a related but completely different concept:

**Negative examples** (showing dog faces, human faces, other animals to a cat detector) are real images of non-cats. You deliberately include them during training so the model learns to say "not cat" correctly. This is a *good* thing — it makes the model more robust. A harder version, **hard negative mining**, selects non-cat images that visually resemble cats to maximally challenge the model during training.

**Adversarial examples** are something completely different and more unsettling:

> Take a cat image the model correctly classifies as "cat" with 99% confidence.  
> Add a tiny, carefully crafted perturbation — invisible to the human eye.  
> The model now outputs "toaster" with 97% confidence.  
> The two images look **identical** to a human.

```
Original cat image      →  model says "cat"     (99% confidence)  ✓
+ imperceptible noise   →  model says "toaster"  (97% confidence)  ✗
```

The perturbation is not random noise. It is computed by finding the direction in ℝ^150,528 that most rapidly crosses the model's decision boundary — then taking the smallest possible step in that direction. The result is a point that is:
- On the manifold (looks like a valid image to humans)
- Just across the decision boundary (the model is completely wrong)
- Extremely close to the original (the difference is invisible)

This works because the model's decision boundaries are not aligned with human perception. In the high-dimensional space, there are directions where a tiny step changes the predicted class dramatically — directions that happen to be perceptually invisible to us.

**Summary of the three concepts:**

| Technique | What it is | Effect on model |
|-----------|-----------|-----------------|
| **Negative examples** | Real images of non-target class | Makes model correctly reject non-targets — improves it |
| **Hard negatives** | Non-target images similar to target | Sharpens decision boundary — improves it |
| **Adversarial examples** | Imperceptibly modified target images | Exploits the geometry of the decision boundary — attacks it |

### More data helps, but never fully solves it

More training samples → denser coverage of the manifold → less extrapolation → better generalisation.

But the manifold is so high-dimensional that you can never fully cover it. This is why foundation models trained on billions of images still fail on edge cases — they are still extrapolating somewhere.

**The connection back to the word source:**

Your 10,000 word trials are a finite sample from the distribution over all 17,576 possible words. The histogram you build is an approximation — it interpolates well for common words, but if a word pattern never appeared in your sample, the model has no information about it. Scale that to 150,528 dimensions and you have the core challenge of deep learning.

---

## 8. Why Mean Alone Is Not Enough

Once you have a distribution, what can you do with it that you cannot do with the mean?

Consider two image patches from the same photograph — both with mean brightness 161. Are they telling the same story?

No. The mean is a single-point summary. The distribution is the **complete description**.

| What you know | What it tells you | What it misses |
|--------------|-------------------|----------------|
| Mean only | The center | Spread, shape, tails, symmetry |
| Mean + std | Center + typical spread | Shape, skewness, tail behaviour |
| Full distribution | Everything | Nothing — it's the complete model |

---

## 9. The Idea vs. The Reality: Distribution vs. Histogram

**Probability Distribution (the blueprint)**
- Theoretical — a model we assume or derive from first principles
- Smooth curve (PDF) or exact probabilities (PMF)

**Data Distribution / Histogram (what you measure)**
- Finite, noisy — what you actually observe from an experiment
- Converges to the true distribution as you collect more data

> The histogram is your approximation. The distribution is what it's approximating.

Knowing how many samples you need for a reliable estimate is a core CV engineering skill.

---

## 10. From Distribution to Generation

If a distribution **describes** the pixel values in a patch, then **sampling from that distribution generates new patches that look like the original**.

This is the seed of generative models. VAEs, GANs, and diffusion models all do exactly this — but with learned, high-dimensional distributions over ℝ^150,528 that preserve spatial structure.

A histogram-sampled patch gets the statistics right but loses all spatial arrangement — because the histogram captures *what* values appear, not *where*. That gap between a 1D histogram and a full spatial model is the gap between Part 0 and modern deep learning. This series builds the foundation for crossing it.

---

## 11. Reading a Distribution: Shape → Physical Meaning

| Shape feature | Mathematical name | CV / image meaning |
|--------------|-------------------|--------------------|
| **Center** | Mean ($\mu$) or mode | Expected pixel value for that material / lighting |
| **Width** | Standard deviation ($\sigma$) | Noise level — variation in a texture patch |
| **Asymmetry** | Skewness | Clipping at 0 or 255 squashes the tail |
| **Heavy tails** | Kurtosis | Rare events — hot pixels, defects, outlier keypoints |
| **Multiple peaks** | Multimodality | Edge region — pixel alternates between two states |

---

## 12. Parameters Compress the Distribution

A distribution can be fully described by a small number of **parameters**.

For the vowel-count example, two numbers fully determine the shape:

$$n = 3 \quad (\text{letters per word}), \qquad p = 5/26 \quad (\text{vowel probability})$$

| Distribution | Parameters | What they control | Info needed |
|-------------|-----------|-------------------|-------------|
| Binomial | $n$, $p$ | number of trials, success prob | 2 numbers |
| Poisson | $\lambda$ | expected count | 1 number |
| Normal | $\mu$, $\sigma$ | center, spread | 2 numbers |
| Neural network | weights W | shape of P(**x**) over ℝ^150,528 | billions of numbers |

Parameters are the **compression algorithm for uncertainty**. A Gaussian noise model needs only μ and σ. A deep generative model needs billions of weights — because the distribution it is compressing is incomparably more complex.

---

## 13. The Complete Ladder

```
Random Process           — press the button / capture a patch / expose a sensor
    ↓
Sample Space Ω           — all 17,576 words / all 256 pixel values / all electron counts
    ↓
Event A ⊆ Ω              — "word has ≥1 vowel" / "pixel is bright" / "count > 50"
    ↓
Random Variable X        — vowel count (scalar) / pixel intensity (scalar) / match distance (scalar)
Random Vector x          — SIFT descriptor (128-d) / image patch (2304-d) / full image (150,528-d)
    ↓
                ┌─────────────────────────────────────────────────┐
                │  For low-dimensional X (scalar or small vector) │
                │  fit distribution directly in observation space  │
                │  → Bernoulli / Binomial / Poisson / Normal       │
                └─────────────────────────────────────────────────┘
                ┌─────────────────────────────────────────────────┐
                │  For high-dimensional x (images, patches)        │
                │  encoder f(x): ℝ^150,528 → ℝ^d  (d << 150,528) │
                │  fit distribution in feature/latent space z ∈ ℝ^d│
                │  → VAE prior N(0,I) / GAN latent / diffusion     │
                └─────────────────────────────────────────────────┘
    ↓
Distribution             — P(X=k) table / Poisson(λ) / Normal(μ,σ) / P(z) = N(0,I) in latent space
    ↓
Parameters               — (n, p) / λ / (μ, σ) / neural network weights W
```

Every probability statement in CV lives somewhere on this ladder. The distributions in Parts 1–6 cover the left branch. Deep learning covers the right branch — but it is built on the same mathematical foundation.

---



---

### Should You Use Adversarial Training in Practice?

**Short answer:** not by default — but worth knowing, especially for industrial inspection.

**Is it common?**
Adversarial training is not standard practice for most CV models. It is used in security-critical applications — autonomous driving, facial recognition, content moderation — where someone might deliberately try to fool the model. For general-purpose CV it is niche.

**Does it make models better?**
It makes models more **robust to adversarial perturbations** — but there is a real tradeoff:

| Property | Standard training | Adversarial training |
|----------|------------------|----------------------|
| Accuracy on clean data | Higher | Slightly lower |
| Robustness to adversarial attacks | Poor | Much better |
| Training time | Normal | 3–10× longer |
| Data needed | Normal | More |

**Is it relevant for industrial inspection?**
Yes — for two reasons:
1. A malicious actor could craft adversarial inputs to hide defects from your inspection system
2. More importantly: natural variations in lighting, camera angle, surface finish, and sensor noise create **natural adversarial examples** — real-world inputs that fall just outside your training distribution. Adversarially robust models tend to be more resilient to these too

If your model is deployed in a safety-critical inspection pipeline, adversarial robustness is worth understanding even if you don't train with adversarial examples explicitly.

---

## References

### Foundational Papers — Adversarial Examples

1. **Intriguing properties of neural networks**
   Szegedy, Zaremba, Sutskever, Bruna, Erhan, Goodfellow, Fergus — *ICLR 2014*
   First paper to discover adversarial examples — small imperceptible perturbations that cause deep networks to misclassify. The starting point for the entire field.

2. **Explaining and Harnessing Adversarial Examples**
   Goodfellow, Shlens, Szegedy — *ICLR 2015*
   Introduced the Fast Gradient Sign Method (FGSM) — a simple one-step attack — and proposed adversarial training as a defence. The most-cited paper in the area.

3. **Towards Deep Learning Models Resistant to Adversarial Attacks**
   Madry, Makelov, Schmidt, Tsipras, Vladu — *ICLR 2018*
   Introduced Projected Gradient Descent (PGD) attack — the gold standard for evaluation — and showed that adversarial training against PGD produces genuinely robust models. The practical foundation for adversarial training.

4. **Towards Evaluating the Robustness of Neural Networks**
   Carlini, Wagner — *IEEE S&P 2017*
   Developed the C&W attack, which defeated most proposed defences at the time. Raised the bar for what "robust" actually means.

5. **Obfuscated Gradients Give a False Sense of Security**
   Athalye, Carlini, Wagner — *ICML 2018*
   Showed that many published defences were not truly robust — they just hid gradients. Essential reading before trusting any robustness claim.

### Suggested Reading Order

If you are new to this: read **2 → 1 → 3** in that order.
- Paper 2 gives you the intuition and the simplest attack
- Paper 1 gives you the original discovery and framing
- Paper 3 gives you the practical defence framework

---

## 14. Why Different Situations Need Different Distributions

| Process | What's random | Key constraint | Distribution |
|---------|---------------|----------------|--------------|
| Pixel above threshold or not | Binary outcome | Only 0 or 1 | **Bernoulli** |
| How many pixels in a patch exceed threshold | Count of successes | Bounded by $n$ | **Binomial** |
| Keypoints or defects in a region | Rare events, large $n$, small $p$ | Mean = variance | **Poisson** |
| Filter output, averaged pixel block | Sum of many terms | Any real value | **Normal** |
| Full natural image | High-dimensional random vector | Lives on a manifold | **Deep generative model** |

**In the next parts** we derive each distribution from first principles — Bernoulli → Binomial → Poisson → Normal — building the mathematical foundation for everything above.
