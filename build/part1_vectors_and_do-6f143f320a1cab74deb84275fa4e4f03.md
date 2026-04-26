# Part 1 — Vectors and Dot Product

## Learning Objective

Understand how image patches become vectors and how the dot product measures
agreement between two patches — the foundation for all normalised template
matching.

---

## Roadmap

| Section | Concept | Why you need it for normalisation |
|---------|---------|-----------------------------------|
| 1 | Pixels as vectors | Template matching compares vectors |
| 2 | Dot product | Foundation for correlation and angle measurement |

---

## 1. Pixels as Vectors

### Intuition

An image patch is a 2D grid of pixel values. But for math, we **flatten** it
into a 1D vector — just a list of numbers. A 3×3 patch with 9 pixels becomes a
point in **9-dimensional space**.

This is not an abstraction — it is literally how OpenCV's template matching
works internally. It flattens the template and each scene patch into vectors,
then compares them.

### Why this matters

Once we see patches as vectors, all of linear algebra applies:

- **Comparing two patches** = measuring the distance or angle between two vectors
- **Brightness change** = adding a vector to shift the point in space
- **Contrast change** = scaling the vector (stretching it longer or shorter)

For visualization, we use **3-pixel patches** (3D space) so we can actually
plot them. Everything generalises to any number of pixels.

---

## 2. Dot Product

### Intuition

The dot product of two vectors answers: **how much do these two vectors point
in the same direction?**

For image patches, the dot product tells you how much two patches "agree" —
pixel by pixel, do they go bright in the same places and dark in the same
places?

### The formula

For two vectors $\vec{a} = [a_1, a_2, \ldots, a_n]$ and
$\vec{b} = [b_1, b_2, \ldots, b_n]$:

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i \cdot b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

Multiply corresponding elements, then sum.

### The geometric definition

The dot product has an equivalent geometric definition:

$$\vec{a} \cdot \vec{b} = \|\vec{a}\| \cdot \|\vec{b}\| \cdot \cos\theta$$

where $\theta$ is the **angle between the two vectors**.

This connects the algebraic formula (multiply and sum) to geometry (angles).
We will use this in Part 2 to derive cosine similarity.

### Key observation

A higher dot product means more agreement between two patches. However, the
**raw dot product also depends on the magnitude** of both vectors — a brighter
patch produces a larger dot product even with the same pattern. Normalising
fixes this (covered in Part 2).
