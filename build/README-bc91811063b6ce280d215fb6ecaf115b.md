# Part I — Math Foundations

The mathematical prerequisites for the rest of the book.

Part I is taught in **three levels** rather than as three separate courses.
The probability and statistics material is deliberately **interleaved** —
you learn enough probability to read real data, spend time with real data,
then go back for the distributions that let you model it, then finish the
statistics toolkit. This is the pedagogical arc Allen Downey's *Think
Stats* gets right and most courses get wrong.

Linear algebra runs as an **independent track** alongside. It's orthogonal
to the prob ↔ stats arc — needed throughout the book, but not gated by
either.

---

## The curriculum map

```
                    ┌─ Level 1 — Foundations ───────────────┐
                    │                                       │
                    │   probability/part0   What is a        │
                    │                       distribution?    │
                    │   probability/part1   Bernoulli        │
                    │        ↓                               │
                    │   statistics/ch01     EDA              │
                    │                       (real data)      │
                    │   statistics/ch02     Histograms       │
                    │                       (shape question) │
                    └───────────────────────────────────────┘
                                     ↓
                    ┌─ Level 2 — Distributions ─────────────┐
                    │                                       │
                    │   probability/part2   Binomial        │
                    │   probability/part3   Poisson         │
                    │   probability/part4   Normal          │
                    │   probability/part5   CLT             │
                    │   probability/part6   Sensor model    │
                    │                                       │
                    │   (we now have the shapes needed to   │
                    │    answer the Ch 2 question)          │
                    └───────────────────────────────────────┘
                                     ↓
                    ┌─ Level 3 — Inference ─────────────────┐
                    │                                       │
                    │   statistics/ch03–ch14                │
                    │   PMF → CDF → modeling → PDF →        │
                    │   relationships → estimation →        │
                    │   hypothesis testing → least squares →│
                    │   regression → time series →          │
                    │   survival → analytic methods         │
                    └───────────────────────────────────────┘

                 ┌─ Linear Algebra (independent track) ─────┐
                 │                                          │
                 │   part1   Vectors and dot product        │
                 │   part2   Norms and similarity           │
                 │   part3   Orthogonality and projection   │
                 │   part4   Linear transforms              │
                 │                                          │
                 │   Read in parallel to any level above.   │
                 └──────────────────────────────────────────┘
```

---

## Why interleaved?

A normal probability-then-statistics course teaches distributions first,
in the abstract, and then drops real data on you in a separate course.
Students arrive at real data armed with formulas they can't connect to
anything concrete, and arrive at distributions not knowing what they're
for.

The interleaved path flips this:

1. **Level 1** builds the minimum probability needed to understand "a
   random variable has an outcome" (parts 0–1), then dives straight into
   real survey data (stats ch01–02). By the end of histograms the reader
   has a concrete question — *"what mathematical shape matches this
   histogram?"* — that cries out for an answer.
2. **Level 2** provides the answer: the family of standard distributions
   (binomial, Poisson, normal, CLT), each introduced with the exact
   shape question it solves.
3. **Level 3** returns to the data with the distributions in hand, and
   builds the full statistical-inference toolkit: PMF, CDF, PDF,
   estimation, hypothesis testing, regression, time series, survival.

Every chapter after Level 1 has a motivating question that came out of
the chapter before. Nothing is introduced "because it's on the syllabus."

---

## How to read this section

- **Sequential (recommended).** Follow the levels top to bottom. Linear
  algebra can be read in any parallel order.
- **Already know probability?** Skip Level 1's probability pieces, skim
  Level 2, read Level 3 in full.
- **Already know statistics?** Use Part I as a refresher. Level 1 is a
  fast read; Level 2 is a review of distribution families; Level 3 has
  the nontrivial material.
- **Need something specific?** Every chapter lists prerequisites at the
  top. Jump in, backfill from earlier levels as needed.

---

## Folder layout

```
math/
├── README.md                 ← this file (curriculum map)
├── probability/              ← Levels 1, 2 (and Level 3 background)
├── statistics/               ← Levels 1, 3 (skip ch02 if taught linearly
│                                — it's in Level 1 above)
└── linear_algebra/           ← independent track
```

The files themselves never moved — the levels are an organization overlay
on top of the same chapter files. The `myst.yml` TOC renders the three
levels; this README is the reference for readers who want to understand
*why* the order is the way it is.
