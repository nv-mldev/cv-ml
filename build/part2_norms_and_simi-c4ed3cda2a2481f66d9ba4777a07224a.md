# Part 2 — Norms and Similarity

## Learning Objective

Understand the L2 norm as a measure of image energy, how normalisation removes
brightness, and why cosine similarity handles contrast scaling but fails on
brightness offsets.

---

## 3. L2 Norm (Vector Length)

### Intuition

The **L2 norm** is the length of a vector — the distance from the origin to the
point. For an image patch, it measures the overall **energy** or **magnitude**
of the pixel values.

$$\|\vec{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2} = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

A bright image has a large L2 norm. A dark image has a small L2 norm. Two
images with the **same pattern but different brightness** have different L2
norms.

### Connection to the dot product

The squared L2 norm is just the dot product of a vector with itself:

$$\|\vec{v}\|^2 = \vec{v} \cdot \vec{v} = \sum v_i^2$$

This is why the L2 norm is also called the **energy** of the signal — it is the
total "power" in the pixel values.

---

## 4. Unit Vectors and Normalisation

### Intuition

A **unit vector** has length 1. Dividing any vector by its L2 norm gives a unit
vector pointing in the **same direction**:

$$\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}$$

For image patches, this operation **removes brightness** (the length) and keeps
only the **pattern** (the direction). Two patches with the same pattern but
different brightness produce the **same unit vector**.

All normalised patches lie on the unit sphere. Patches with the same pattern
map to the same point on that sphere regardless of brightness.

---

## 5. Cosine Similarity

### Intuition

Now we can define a proper **similarity measure** that ignores brightness. From
the geometric definition of the dot product:

$$\vec{a} \cdot \vec{b} = \|\vec{a}\| \cdot \|\vec{b}\| \cdot \cos\theta$$

Rearranging:

$$\cos\theta = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \cdot \|\vec{b}\|}$$

This is the **cosine similarity** — the dot product of the two unit vectors. It
measures the angle between them, ignoring their lengths (brightness).

| $\cos\theta$ | Meaning | Image example |
|---------------|---------|---------------|
| $+1$ | Same direction | Same pattern, any brightness |
| $0$ | Perpendicular | Completely unrelated patterns |
| $-1$ | Opposite direction | Inverted pattern (negative image) |

### The blind spot: brightness offset

Cosine similarity handles **contrast scaling** ($a \cdot T$) perfectly because
scaling changes the vector length but not its direction.

However, a **brightness offset** ($T + b$) shifts every pixel by the same
constant. This pushes the vector toward the $[1, 1, \ldots, 1]$ direction,
changing the angle — so cosine similarity gives the wrong answer.

This is why `TM_SQDIFF_NORMED` fails on scenes with illumination offsets.
Fixing it requires mean subtraction, which is covered in Part 3.
