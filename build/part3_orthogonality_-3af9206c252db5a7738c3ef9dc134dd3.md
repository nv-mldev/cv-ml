# Part 3 — Orthogonality and Projection

## Learning Objective

Understand orthogonality as the independence of two vector directions, and
show that mean subtraction is exactly orthogonal projection onto the
$[1, 1, \ldots, 1]$ brightness direction — making brightness and pattern
independent components.

---

## 6. Orthogonality

### Intuition

Two vectors are **orthogonal** (perpendicular) when their dot product is zero:

$$\vec{a} \cdot \vec{b} = 0 \quad \Longleftrightarrow \quad \vec{a} \perp \vec{b}$$

Geometrically, they point in completely independent directions — knowing one
tells you nothing about the other.

**Why this matters for normalisation:** Mean subtraction splits a patch vector
into two orthogonal parts — a brightness component and a pattern component.
Because they are orthogonal, changing brightness **cannot** affect the pattern.
That independence is the whole reason normalisation works.

### The key orthogonality we need

The orthogonality that matters for normalisation is between:

- The **uniform brightness vector** $[1, 1, 1, \ldots, 1]$ — equal value
  everywhere, encodes only brightness, zero spatial pattern
- The **mean-subtracted residual** of any vector

Any scalar multiple of $[1, 1, \ldots, 1]$ is a flat gray image. Mean
subtraction removes exactly this component.

---

## 7. Orthogonal Projection and Decomposition

### Intuition

**Orthogonal projection** splits a vector into two perpendicular parts:

$$\vec{v} = \underbrace{\text{proj}_{\vec{u}} \vec{v}}_{\text{component along } \vec{u}} + \underbrace{(\vec{v} - \text{proj}_{\vec{u}} \vec{v})}_{\text{component perpendicular to } \vec{u}}$$

The projection formula onto a unit vector $\hat{u}$:

$$\text{proj}_{\hat{u}} \vec{v} = (\vec{v} \cdot \hat{u})\, \hat{u}$$

### Mean subtraction is orthogonal projection

When $\vec{u} = [1, 1, \ldots, 1]$ (the uniform brightness direction), the
projection simplifies to:

$$\text{proj} = \bar{v} \cdot [1, 1, \ldots, 1]$$

That is just the **mean** of the pixel values, broadcast to every element. So
**mean subtraction IS orthogonal projection** — it removes the component along
$[1, 1, \ldots, 1]$.

### The projection formula derivation

Let $\vec{u} = [1, 1, \ldots, 1]$ with $n$ elements. Its unit vector is:

$$\hat{u} = \frac{[1,1,\ldots,1]}{\sqrt{n}}$$

The projection of $\vec{v}$ onto $\hat{u}$:

$$\text{proj}_{\hat{u}} \vec{v}
= (\vec{v} \cdot \hat{u})\,\hat{u}
= \left(\frac{\sum v_i}{\sqrt{n}}\right) \cdot \frac{[1,1,\ldots,1]}{\sqrt{n}}
= \frac{\sum v_i}{n} \cdot [1,1,\ldots,1]
= \bar{v} \cdot [1,1,\ldots,1]$$

So the projection is just the **mean** broadcast to every element. Mean
subtraction removes this projection, leaving only the perpendicular (pattern)
component.

### Key consequence

A brightness offset only changes the brightness component, not the pattern
component. After mean subtraction, patterns are identical regardless of uniform
lighting shifts.
