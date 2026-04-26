# Contents

A reading map of the whole book. Page-level navigation lives in the
sidebar; this page is for stepping back and seeing the arc.

---

## [Preface](preface.md)
Why this book exists, and three suggested reading paths.

---

## [Chapter 0 — From Measurements to Meaning](ch00_introduction.md)
The clean world, mathematical modelling, where noise comes from, why it's
Gaussian, the inverse problem, and the three attacks (signal processing /
parametric fitting / machine learning) that organise the rest of the book.

---

## Part I — Math Foundations

Probability, statistics, and linear algebra developed at the level needed
for the rest of the book. Three levels, plus an independent linear-algebra
track. See [Part I overview](../../math/README.md).

### Level 1 — Foundations (interleaved probability ↔ statistics)
- [Probability — overview](../../math/probability/README.md)
- [What is a distribution?](../../math/probability/part0_what_is_a_distribution.md)
- [Bernoulli](../../math/probability/part1_bernoulli.md)
- [Statistics — overview](../../math/statistics/README.md)
- [Ch 1 — Exploratory data analysis](../../math/statistics/ch01_eda.md)
- [Ch 2 — Distributions](../../math/statistics/ch02_distributions.md)

### Level 2 — Distributions (probability deep dive)
- [Binomial](../../math/probability/part2_binomial.md)
- [Poisson](../../math/probability/part3_poisson.md)
- [Normal](../../math/probability/part4_normal.md)
- [Central Limit Theorem](../../math/probability/part5_clt.md)
- [Putting it together](../../math/probability/part6_putting_it_together.md)
- [Exercises](../../math/probability/exercises.md)
- [Applied — From photons to pixel noise](../../math/probability/applied_sensors.md)
  *(capstone — applies the distribution chain to the sensor model)*

### Level 3 — Inference (statistics deep dive)
- [Ch 3 — PMFs](../../math/statistics/ch03_pmf.md)
- [Ch 4 — CDFs](../../math/statistics/ch04_cdf.md)
- [Ch 5 — Modelling](../../math/statistics/ch05_modeling.md)
- [Ch 6 — PDFs](../../math/statistics/ch06_pdf.md)
- [Ch 7 — Relationships between variables](../../math/statistics/ch07_relationships.md)
- [Ch 8 — Estimation](../../math/statistics/ch08_estimation.md)
- [Ch 9 — Hypothesis testing](../../math/statistics/ch09_hypothesis_testing.md)
- [Ch 10 — Least squares](../../math/statistics/ch10_least_squares.md)
- [Ch 11 — Regression](../../math/statistics/ch11_regression.md)
- [Ch 12 — Time series](../../math/statistics/ch12_time_series.md)
- [Ch 13 — Survival analysis](../../math/statistics/ch13_survival.md)
- [Ch 14 — Analytic methods](../../math/statistics/ch14_analytic_methods.md)
- [Exercises](../../math/statistics/exercises.md)

### Linear Algebra (independent track)
- [Linear algebra — overview](../../math/linear_algebra/README.md)
- [Part 1 — Vectors and the dot product](../../math/linear_algebra/part1_vectors_and_dot_product.md)
- [Part 2 — Norms and similarity](../../math/linear_algebra/part2_norms_and_similarity.md)
- [Part 3 — Orthogonality and projection](../../math/linear_algebra/part3_orthogonality_and_projection.md)
- [Part 4 — Linear transforms](../../math/linear_algebra/part4_linear_transforms.md)
- [Exercises](../../math/linear_algebra/exercises.md)
- [Applied — Images as vectors, patches as points](../../math/linear_algebra/applied_images.md)
  *(capstone — applies the linear algebra to image-patch comparison)*

---

## Part II — Signals and Measurement
How a continuous physical scene becomes a finite grid of pixel numbers.

- [Ch 1 — Digitisation: sampling, Nyquist, aliasing](../part2_signals_and_measurement/ch01_digitisation/ch01_digitisation.md)
- [Ch 2 — The sensor](../part2_signals_and_measurement/ch02_sensor/ch02_sensor.md)
- [Ch 3 — Pixels](../part2_signals_and_measurement/ch03_pixels/ch03_pixels.md)
- [Ch 4 — Contrast](../part2_signals_and_measurement/ch04_contrast/ch04_contrast.md)
- [Ch 5 — Colour](../part2_signals_and_measurement/ch05_colour/ch05_colour.md)

---

## Part III — Why Raw Signals Fail
What pixels alone can and cannot tell you, and why we need higher-level
representations.

- [Ch 6 — Pixel problems](../part3_why_raw_signals_fail/ch06_pixel_problems/ch06_pixel_problems.md)

---

## Part IV — Learning from Signals
From convolutions to trained networks.

- [Ch 9 — Convolutions](../part5_learning_from_signals/ch09_convolutions/ch09_convolutions.md)
- [Ch 10 — Backpropagation](../part5_learning_from_signals/ch10_backprop/ch10_backprop.md)
- [Ch 11 — CNNs](../part5_learning_from_signals/ch11_cnns/ch11_cnns.md)
- [Ch 12 — Training](../part5_learning_from_signals/ch12_training/ch12_training.md)

---

## Part V — Attention and Beyond
Self-attention, vision transformers, and multimodal models.

- [Ch 13 — Attention](../part6_attention_and_beyond/ch13_attention/ch13_attention.md)
- [Ch 14 — Vision Transformer](../part6_attention_and_beyond/ch14_vit/ch14_vit.md)
- [Ch 15 — Vision-Language Models](../part6_attention_and_beyond/ch15_vlm/ch15_vlm.md)

---

**Start →** [Chapter 0 — From Measurements to Meaning](ch00_introduction.md)
