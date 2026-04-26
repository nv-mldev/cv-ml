# Chapter 8 — Linear Algebra for Images
### Images as Vectors, Patches as Points

> *Chapter 6 introduced L2 normalisation and mean subtraction as tools for invariant matching. This chapter explains the geometry behind those operations — and reveals that Pearson correlation is just a dot product between unit-mean vectors.*

---

## 8.1 An Image Patch Is a Vector

A $3 \times 3$ grayscale patch has 9 pixel values. Unroll them into a column vector:

$$\mathbf{x} = [x_1, x_2, \ldots, x_9]^\top \in \mathbb{R}^9$$

This is a **point in 9-dimensional space**. Every possible $3 \times 3$ patch is a different point. Comparing two patches means measuring the distance or angle between two points in this space.

For an $m \times n$ patch: $\mathbf{x} \in \mathbb{R}^{mn}$. High-resolution patches live in high-dimensional spaces — a $64 \times 64$ patch is a point in $\mathbb{R}^{4096}$.

---

## 8.2 Dot Product — Measuring Agreement

$$\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$$

The dot product measures pixel-by-pixel agreement. But it depends on the magnitude of both vectors — a brighter patch has a larger dot product with everything, regardless of shape similarity.

---

## 8.3 L2 Norm and Unit Vectors

$$\|\mathbf{x}\| = \sqrt{\sum_i x_i^2}$$

Dividing by the norm gives a **unit vector** $\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$ with $\|\hat{\mathbf{x}}\| = 1$.

All unit vectors lie on the surface of the unit hypersphere. Their dot product is:

$$\hat{\mathbf{x}} \cdot \hat{\mathbf{y}} = \cos\theta$$

where $\theta$ is the angle between them. **Cosine similarity** measures the angle — independent of magnitude. A contrast change ($\mathbf{y} = a\mathbf{x}$) does not change the angle, so it does not change cosine similarity.

This is the geometric interpretation of L2 normalisation from Chapter 6.

---

## 8.4 Mean Subtraction — Projecting Out the Brightness Direction

The vector $\mathbf{1} = [1, 1, \ldots, 1]^\top$ points in the "uniform brightness" direction. Projecting $\mathbf{x}$ onto $\mathbf{1}$ gives the mean; subtracting it removes the mean:

$$\tilde{\mathbf{x}} = \mathbf{x} - \bar{x}\mathbf{1}$$

Geometrically, mean subtraction projects $\mathbf{x}$ onto the hyperplane orthogonal to $\mathbf{1}$. A brightness offset ($\mathbf{y} = \mathbf{x} + b\mathbf{1}$) adds a component along $\mathbf{1}$ — subtracted mean removes it.

After mean subtraction, the dot product is unaffected by uniform brightness changes. This is the geometric interpretation of mean subtraction from Chapter 6.

---

## 8.5 Pearson Correlation = Cosine of Mean-Subtracted Vectors

Combine both operations:

$$r(\mathbf{x}, \mathbf{y}) = \frac{\tilde{\mathbf{x}} \cdot \tilde{\mathbf{y}}}{\|\tilde{\mathbf{x}}\| \|\tilde{\mathbf{y}}\|} = \cos\theta_{\tilde{\mathbf{x}}, \tilde{\mathbf{y}}}$$

Pearson correlation is the **cosine similarity of mean-subtracted vectors**. It is invariant to any $aI + b$ transform because:
- Mean subtraction removes $b$ (projects out $\mathbf{1}$ component)
- L2 normalisation removes $a$ (cancels magnitude)

This geometric view makes the invariance obvious — and makes the ceiling obvious: once the pixel grid shifts (rotation, scale), the vectors $\tilde{\mathbf{x}}$ and $\tilde{\mathbf{y}}$ have different components shuffled, and no amount of normalisation restores alignment.

---

## 8.6 Orthogonality and Transforms

Two vectors are **orthogonal** when $\mathbf{x} \cdot \mathbf{y} = 0$ — they are geometrically perpendicular, carrying completely independent information.

An **orthogonal transform** $Q$ preserves norms and dot products:

$$\|Q\mathbf{x}\| = \|\mathbf{x}\|, \quad (Q\mathbf{x}) \cdot (Q\mathbf{y}) = \mathbf{x} \cdot \mathbf{y}$$

The Fourier transform is an orthogonal transform — it decomposes an image into orthogonal frequency components (the basis of Chapter 1's frequency analysis) while preserving energy (Parseval's theorem).

---

## 8.7 The Manifold Hypothesis — Why High-Dimensional Pixel Space Is Nearly Empty

An $m \times n$ image is a point in $\mathbb{R}^{mn}$. For a $64 \times 64$ image that is $\mathbb{R}^{4096}$. The number of possible images is $256^{4096}$ — astronomically large.

But **natural images** occupy a tiny, thin slice of this space. Most points in $\mathbb{R}^{4096}$ are random noise — not images of anything real. The set of natural images forms a low-dimensional **manifold** embedded in the high-dimensional pixel space.

This is the **manifold hypothesis**, and it explains why learned features work: CNNs learn to map the high-dimensional pixel space to a lower-dimensional representation that captures where you are on the natural image manifold — not where you are in the raw pixel cube.

> **Simulation:** `~/projects/cv-ml/math/linear_algebra/`
> — All 4 parts: vectors and dot product → norms and similarity → orthogonality → transforms.

---

## Summary

| Concept | Geometric meaning | Connection to Ch 6 |
|---------|-------------------|-------------------|
| Pixel patch as vector | Point in $\mathbb{R}^{mn}$ | Enables geometric comparison |
| Dot product | Pixel-by-pixel agreement; magnitude-dependent | Raw SSD without normalisation |
| L2 norm | Vector length | Divisor in L2 normalisation |
| Cosine similarity | Angle between vectors; magnitude-independent | Handles contrast ($a$) |
| Mean subtraction | Project out $\mathbf{1}$ component | Handles brightness offset ($b$) |
| Pearson correlation | $\cos\theta$ of mean-subtracted vectors | Full affine invariance |
| Manifold hypothesis | Natural images = thin slice of pixel space | Motivates learned features |

---

**Next →** [Chapter 9 — Convolutions and Filtering](../../part5_learning_from_signals/ch09_convolutions/README.md): Part V begins — moving from comparing patches to learning features that are invariant to spatial transforms.
