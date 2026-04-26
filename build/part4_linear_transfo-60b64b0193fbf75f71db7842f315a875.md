# Part 4 — Linear Transforms

## Learning Objective

Understand the lighting model $I = aT + b$ as a general linear transform, see
what each parameter does geometrically in pixel-value space, and learn why
orthogonal transforms (DFT, DCT) preserve energy while general linear
transforms do not.

---

## 8. Linear Transformations

### Intuition

In real-world template matching, the scene patch is rarely a pixel-perfect copy
of the template. Lighting changes. The relationship is modelled as a **linear
transform**:

$$I = a \cdot T + b$$

where:

- $a$ = **contrast change** (stretches/compresses the pixel spread)
- $b$ = **brightness offset** (shifts all pixels up or down uniformly)

### Geometric effect in pixel space

**Scaling ($a$):** Multiplies the vector by a scalar. Changes the **length**
(distance from origin) but not the **direction**. All points stay on the same
ray through the origin. Cosine similarity handles this correctly.

**Offset ($b$):** Adds $b \cdot [1,1,\ldots,1]$ to the vector. Shifts the
point **toward the $[1,1,1]$ diagonal**. This changes the direction — so cosine
similarity gives the wrong answer. Mean subtraction is needed to undo this.

---

## 9. Orthogonal (Energy-Preserving) Transforms

### Intuition

An **orthogonal transform** is a special linear transform where the matrix $Q$
satisfies:

$$Q^\top Q = I \quad \text{(identity matrix)}$$

This means:

- The columns of $Q$ are orthonormal (perpendicular unit vectors)
- The transform is a pure **rotation** (and/or reflection) — no stretching
- **Energy is preserved:** $\|Qx\|^2 = \|x\|^2$

A general linear transform $Y = AX$ can change the L2 norm (energy) of the
signal:

$$\|AX\| \neq \|X\| \quad \text{in general}$$

This means the transform distorts the signal — it stretches some directions and
compresses others.

### Parseval's theorem

Transforms like the **DFT (Discrete Fourier Transform)** and **DCT** are
orthogonal. When you transform a signal to the frequency domain, the total
energy is the same — it is just redistributed across frequency bins instead of
pixel positions. This is **Parseval's theorem**:

$$\sum_{n} |x[n]|^2 = \frac{1}{N} \sum_{k} |X[k]|^2$$

### The hierarchy

```
All transforms
  └── Linear transforms  (T(ax + by) = aT(x) + bT(y))
        ├── General linear  (energy changes)
        │     • contrast scaling:  I = 2T
        │     • brightness offset: I = T + 50
        │     • both:              I = aT + b
        │
        └── Orthogonal / Unitary  (energy preserved: ‖Qx‖ = ‖x‖)
              • Rotation matrices
              • DFT (Discrete Fourier Transform)
              • DCT (Discrete Cosine Transform)
              • Hadamard transform
```

**Linearity** is necessary but not sufficient for energy preservation.
**Orthogonality** is the additional property that guarantees it.
