# Part 3: The Poisson Distribution — The Limit of Many Rare Events

## Intuition

The Binomial works perfectly when we know $n$ and $p$ separately. But many CV counting problems involve events that are **rare across a large region** — and the Poisson distribution is the natural model for these.

**CV examples of Poisson processes:**

- **Defect detection:** a manufactured surface has a low defect probability per pixel, but millions of pixels → defect count is Poisson
- **Keypoint detection:** SIFT keypoints are rare in homogeneous regions → count in a patch is Poisson
- **Background modeling:** in a static scene, the number of pixels that change due to noise in any frame is Poisson

In sensor terms: we don't actually know $n$ and $p$ separately. A light source emits an enormous number of photons; each has a tiny probability of reaching our specific photosite. The **Poisson distribution** is what the Binomial becomes when:
- $n \to \infty$ (enormous number of potential events)
- $p \to 0$ (tiny probability each one occurs)
- But $\lambda = np$ stays constant (the expected count is fixed by the physics)

---

## The Derivation

Starting from the Binomial PMF and taking the limit:

$$P(k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Substitute $p = \lambda / n$:

$$P(k) = \binom{n}{k} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

### Step-by-step limit

Expand the binomial coefficient:

$$\binom{n}{k} = \frac{n!}{k!(n-k)!} = \frac{n(n-1)(n-2)\cdots(n-k+1)}{k!}$$

For large $n$, each factor $n, (n-1), (n-2), \ldots$ is approximately $n$, so:

$$\binom{n}{k} \approx \frac{n^k}{k!}$$

Now substitute back:

$$P(k) \approx \frac{n^k}{k!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1 - \frac{\lambda}{n}\right)^{n-k}$$

The $n^k$ terms cancel:

$$P(k) \approx \frac{\lambda^k}{k!} \cdot \left(1 - \frac{\lambda}{n}\right)^{n} \cdot \left(1 - \frac{\lambda}{n}\right)^{-k}$$

As $n \to \infty$:
- $\left(1 - \frac{\lambda}{n}\right)^{n} \to e^{-\lambda}$ (this is the definition of $e$!)
- $\left(1 - \frac{\lambda}{n}\right)^{-k} \to 1$ (since $k$ is fixed and $\lambda/n \to 0$)

$$\boxed{P(k \mid \lambda) = \frac{\lambda^k \, e^{-\lambda}}{k!}}$$

**Mean:** $E[k] = \lambda$
**Variance:** $\text{Var}(k) = \lambda$

The magical property: **the mean equals the variance**. This single fact determines the noise behaviour of every camera sensor.

---

## Why Poisson is the Right Model for Photon Counting

In reality, the number of "available" photons is astronomically large. A typical LED emits ~$10^{18}$ photons per second. The probability that any specific photon reaches our 6µm × 6µm photosite is vanishingly small. But the product $\lambda = np$ (the expected count) is determined by the illumination, reflectance, exposure time, and sensor area — typically tens to thousands of photons.

This is exactly the Poisson regime: enormous $n$, tiny $p$, moderate $\lambda$.

**That's why we use $\text{Poisson}(\lambda)$ for shot noise — not because it's an approximation, but because it's the physically correct model for photon counting.**

### Other CV Processes that are Poisson

The same regime (large $n$, small $p$, fixed $\lambda = np$) describes many counting problems in image analysis:

| Process | Large $n$ | Small $p$ | $\lambda$ | Application |
|---------|-----------|-----------|-----------|-------------|
| Defect detection | Pixels per image | Defect probability per pixel | Defects per m² | Surface inspection |
| Keypoint detection | Pixels in patch | Keypoint probability per pixel | Keypoints per patch | SIFT, ORB matching |
| Background subtraction | Frames | P(pixel changes due to noise) | Changed pixels per frame | Video surveillance |
| Photon counting | Potential photons | P(photon hits photosite) | Expected electrons | Camera sensor |

**Any counting process where: events are rare, the region is large, and events are independent — that count is Poisson.**

---

## The Poisson Distribution Has ONE Parameter

Compare with the Binomial which has two ($n$ and $p$):

| Distribution | Parameters | Mean | Variance | Mean = Variance? |
|-------------|-----------|------|----------|-------------------|
| Binomial($n$, $p$) | $n$, $p$ | $np$ | $np(1-p)$ | Only if $p \to 0$ |
| Poisson($\lambda$) | $\lambda$ | $\lambda$ | $\lambda$ | **Always** |

The Poisson's mean-equals-variance property is incredibly useful. If you measure the mean of a Poisson process, you immediately know the variance (and therefore the noise level). This is why sensor engineers can predict noise from signal level alone.
