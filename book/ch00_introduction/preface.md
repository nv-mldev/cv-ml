# Preface

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

## How to Read This Book

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

**Next →** [Contents](contents.md) for the full reading map, or jump
straight to [Chapter 0 — From Measurements to Meaning](ch00_introduction.md)
to start the book.
