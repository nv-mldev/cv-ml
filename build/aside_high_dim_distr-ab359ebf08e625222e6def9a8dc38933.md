# Aside — High-Dimensional Distributions, Manifolds, and Generative Models

> **This is a forward-reference, not part of the main thread.**
>
> The probability series (parts 1–6 + applied_sensors) builds one-dimensional distributions — Bernoulli, Binomial, Poisson, Normal — and the CLT that glues them together. Everything stays in $\mathbb{R}^1$.
>
> This file collects the deep-end material that needs machinery the main thread doesn't provide yet: random vectors, encoders, neural networks, deep generative models. It's the content that used to live inside `part0` but didn't belong there because it required the reader to already understand things they only meet much later.
>
> **Read order suggestion:** finish parts 1–6 and `applied_sensors.md` first. Come back here once you've met random vectors and started thinking about images / audio clips / spectrograms as high-dimensional objects. This material will eventually be relocated to a dedicated chapter in Part IV, after the CNN and feature-learning material is in place.

---

## 1. Distributions Over High-Dimensional Spaces

Consider a full colour image fed into a neural network. Its dimensions are:

$$3 \times 224 \times 224 = 150{,}528 \text{ dimensions}$$

— where **3** is the number of colour channels (R, G, B) and **224×224** is the spatial size of the image.

So a single image is one point in $\mathbb{R}^{150{,}528}$. The true distribution $P(\mathbf{x})$ over this space has several brutal properties:

### It is wildly multimodal

There is one cluster (mode) for cat images, another for car images, another for faces, another for aerial views — thousands of modes for all visual concepts that exist in the world. The distribution is not a smooth bell curve. It is a jagged, complex, high-dimensional landscape.

### It lives on a low-dimensional manifold

Most random points in $\mathbb{R}^{150{,}528}$ look like pure static noise — they are not valid images. The set of all "natural images" occupies a tiny, curved **manifold** embedded inside that vast space. All real photographs live near this manifold. Everything off it is meaningless.

### You can never fully sample it

The number of possible distinct images is incomprehensibly large. Even if every atom in the universe stored one image, you would cover an infinitesimal fraction of the manifold. So the training dataset is always a sparse, finite sample from the true distribution.

**The scale of the problem:**

| Scale | Distribution | Model type | How you represent it |
|---|---|---|---|
| 1 pixel | $P(x), x \in \mathbb{R}^1$ | **Statistical** | Histogram, Poisson, Gaussian — 1 or 2 parameters |
| 1 patch mean | $P(x), x \in \mathbb{R}^1$ | **Statistical** | Gaussian noise model — just $\mu$ and $\sigma$, no learning |
| SIFT descriptor | $P(\mathbf{x}), \mathbf{x} \in \mathbb{R}^{128}$ | Statistical or shallow ML | GMM, PCA — too complex for a simple histogram |
| Full image | $P(\mathbf{x}), \mathbf{x} \in \mathbb{R}^{150{,}528}$ | **Deep learning** | Neural network — billions of parameters, learned from data |
| Conditional | $P(\mathbf{x} \mid \text{"cat"})$ | **Deep learning** | Conditional generative model (GAN, VAE, diffusion) |

> **Note on "statistical model" vs "deep learning model":** A statistical model is a hand-crafted parametric assumption — you choose the distribution family (Gaussian, Poisson) and estimate its parameters from data. No architecture, no backpropagation. A deep learning model *learns* the distribution family itself from data — the architecture is flexible enough to approximate any distribution given enough parameters and data. Both are models of a distribution. The difference is in how expressive and how automated the representation is.

The math is identical at every scale — it is still just a probability distribution. Only the **representation** of that distribution changes.

---

## 2. The Network Learns in Feature Space, Not Pixel Space

This is a crucial precision that most introductions skip.

When we say "the network learns the distribution over images", we do not mean it learns $P(\mathbf{x})$ directly in the 150,528-dimensional pixel space. What it actually learns is a distribution over a **learned, low-dimensional feature space** — a compressed representation that the network itself constructs through its layers.

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

So the manifold — the curved surface on which all valid cat images sit — mathematically exists inside $\mathbb{R}^{150{,}528}$. But that surface has no clean shape in pixel coordinates: it is twisted, folded, and its distances do not match human semantic similarity. Fitting a probability distribution to a shape like that is intractable.

The manifold of natural images exists in pixel space, but it has no clean parametric form there. Fitting a distribution directly in pixel coordinates is intractable.

### Feature space fixes this

The network learns an encoder mapping:

$$f : \mathbb{R}^{150528} \rightarrow \mathbb{R}^d$$

such that:
- Semantically **similar** images map to **nearby points** in $\mathbb{R}^d$
- Semantically **different** images map to **distant points**
- The distribution $P(\mathbf{z})$ over the feature space is far simpler and more structured than $P(\mathbf{x})$ over pixel space

This is why every major generative architecture works in feature/latent space:

| Architecture | Where the distribution lives | What the distribution looks like |
|---|---|---|
| **VAE** | Latent space $\mathbb{R}^d$ | Gaussian — $P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| **GAN** | Latent space $\mathbb{R}^d$ | Simple prior (Gaussian or Uniform) mapped to images |
| **Stable Diffusion** | Latent space 64×64×4 | Gaussian noise progressively denoised |
| **Contrastive (SimCLR)** | Feature space $\mathbb{R}^d$ | Positives clustered, negatives separated |

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

Everything in parts 1–6 — Bernoulli, Binomial, Poisson, Normal — lives in $\mathbb{R}^1$. One random variable, one number.

The step to deep learning is:
1. Stack many random variables into a random vector → $\mathbb{R}^d$
2. Learn an encoder that maps pixel space to a feature space where the distribution is simple
3. Fit a tractable distribution in that feature space — often approximately Gaussian

This is why the Normal distribution from Part 4 is not just a historical curiosity. It is the **literal prior** used in VAEs:

$$P(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The whole 1-D series is building the mathematical foundation for that one line.

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

The code in `part0_what_is_a_distribution.py` generates a local version of the Swiss roll manifold visualization — no internet connection required.

---

## 3. What Neural Networks Actually Learn — Interpolation and Extrapolation

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

## 4. Distribution Shift — The World Moves, Not the Model

Distribution shift happens when the training distribution $P_{\text{train}}(\mathbf{x})$ and the deployment distribution $P_{\text{deploy}}(\mathbf{x})$ are different. The network learned a good approximation of $P_{\text{train}}$ — but at deployment it receives inputs from $P_{\text{deploy}}$, a region it has never seen.

The model is not wrong. It learned exactly what it was shown. The *world moved*.

| Trained on | Deployed on | What shifts |
|---|---|---|
| Daytime road images | Night-time driving | Lighting distribution |
| Western hospital CT scans | Asian hospital CT scans | Scanner type + patient demographics |
| Clean studio photos | Blurry phone camera photos | Image quality distribution |
| Summer dataset | Winter dataset | Colour, texture, scene distribution |
| Vibration sensor on a healthy motor | Same sensor on a degrading motor | Underlying noise distribution |

In every case the model is extrapolating into a region it has never covered. It may still produce a confident-looking output — but that confidence is not trustworthy. This is one of the hardest real-world problems in deployment.

---

## 5. Adversarial Examples — Not the Same as Negative Examples

Before explaining adversarial examples, distinguish them from a related but completely different concept:

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

The perturbation is not random noise. It is computed by finding the direction in $\mathbb{R}^{150{,}528}$ that most rapidly crosses the model's decision boundary — then taking the smallest possible step in that direction. The result is a point that is:
- On the manifold (looks like a valid image to humans)
- Just across the decision boundary (the model is completely wrong)
- Extremely close to the original (the difference is invisible)

This works because the model's decision boundaries are not aligned with human perception. In the high-dimensional space, there are directions where a tiny step changes the predicted class dramatically — directions that happen to be perceptually invisible to us.

**Summary of the three concepts:**

| Technique | What it is | Effect on model |
|---|---|---|
| **Negative examples** | Real images of non-target class | Makes model correctly reject non-targets — improves it |
| **Hard negatives** | Non-target images similar to target | Sharpens decision boundary — improves it |
| **Adversarial examples** | Imperceptibly modified target images | Exploits the geometry of the decision boundary — attacks it |

### More data helps, but never fully solves it

More training samples → denser coverage of the manifold → less extrapolation → better generalisation.

But the manifold is so high-dimensional that you can never fully cover it. This is why foundation models trained on billions of images still fail on edge cases — they are still extrapolating somewhere.

---

## 6. From Distribution to Generation

If a distribution **describes** the values of a signal, then **sampling from that distribution generates new signals that look like the original**.

This is the seed of generative models. VAEs, GANs, and diffusion models all do exactly this — but with learned, high-dimensional distributions over $\mathbb{R}^{150{,}528}$ (or whatever the data space is) that preserve spatial / temporal structure.

A 1-D histogram-sampled patch gets the marginal pixel statistics right but loses all spatial arrangement — because the histogram captures *what* values appear, not *where*. That gap between a 1-D histogram and a full spatial model is the gap between parts 1–6 of the probability series and modern deep generative modelling. Crossing it is exactly what the rest of the book is about.

---

## 7. Should You Use Adversarial Training in Practice?

**Short answer:** not by default — but worth knowing, especially for industrial inspection.

**Is it common?**
Adversarial training is not standard practice for most CV models. It is used in security-critical applications — autonomous driving, facial recognition, content moderation — where someone might deliberately try to fool the model. For general-purpose CV it is niche.

**Does it make models better?**
It makes models more **robust to adversarial perturbations** — but there is a real tradeoff:

| Property | Standard training | Adversarial training |
|---|---|---|
| Accuracy on clean data | Higher | Slightly lower |
| Robustness to adversarial attacks | Poor | Much better |
| Training time | Normal | 3–10× longer |
| Data needed | Normal | More |

**Is it relevant for industrial inspection?**
Yes — for two reasons:
1. A malicious actor could craft adversarial inputs to hide defects from your inspection system.
2. More importantly: natural variations in lighting, camera angle, surface finish, and sensor noise create **natural adversarial examples** — real-world inputs that fall just outside your training distribution. Adversarially robust models tend to be more resilient to these too.

If your model is deployed in a safety-critical inspection pipeline, adversarial robustness is worth understanding even if you don't train with adversarial examples explicitly.

---

## References — Adversarial Examples

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
