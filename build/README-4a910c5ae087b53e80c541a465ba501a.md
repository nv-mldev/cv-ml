# Linear Algebra for Matching

The linear algebra prerequisites for template matching, feature comparison,
and everything that follows in CV and ML. Every concept is grounded in a
concrete imaging problem: if you understand this series, you understand
why OpenCV's `TM_CCOEFF_NORMED` works the way it does — and by extension,
the math underlying every "compare two patches" operation in CV and ML.

**Master objective:** build up vectors → dot products → norms → cosine
similarity → orthogonality → projections → linear transforms, with each
step motivated by a concrete imaging scenario.

## Parts

| File pair | Topic |
|-----------|-------|
| [`part1_vectors_and_dot_product`](part1_vectors_and_dot_product.md) | Pixels as vectors; dot product definition and geometric meaning. |
| [`part2_norms_and_similarity`](part2_norms_and_similarity.md) | L2 norm, unit vectors, cosine similarity, and the brightness-offset blind spot. |
| [`part3_orthogonality_and_projection`](part3_orthogonality_and_projection.md) | Orthogonality; mean subtraction as orthogonal projection; signal decomposition. |
| [`part4_linear_transforms`](part4_linear_transforms.md) | Lighting model $I = aT + b$; orthogonal transforms; DFT; Parseval's theorem. |
| [`exercises`](exercises.md) | Three practice exercises tying it all together. |

## Running

Every `.py` file is standalone:

```bash
# from project root
source .venv/bin/activate
python math/linear_algebra/part1_vectors_and_dot_product.py
python math/linear_algebra/part2_norms_and_similarity.py
python math/linear_algebra/part3_orthogonality_and_projection.py
python math/linear_algebra/part4_linear_transforms.py
python math/linear_algebra/exercises.py
```

## Concept map

```
Image patch
  └── flatten → n-dimensional vector
        ├── L2 norm              → energy / brightness
        ├── unit vector          → pattern (brightness removed)
        ├── dot product          → pixel-by-pixel agreement
        └── cosine similarity    → angle between patterns
              ├── handles contrast scaling ✓
              └── fails on brightness offset ✗
                    └── fix: mean subtraction
                          = orthogonal projection onto [1,1,...,1]
                          = brightness component removed
                          = pattern component preserved
```

## Prerequisites

- Basic Python and NumPy.
- [`math/probability/`](../probability/README.md) — Parts 0–5 are useful but
  not strictly required.

## Who links here

- [`applied_images.md`](applied_images.md) — the applied capstone of
  this series; treats image patches as vectors and derives the geometric
  meaning of L2 normalisation, mean subtraction, and Pearson correlation.
- `nn-basics/fundamentals/math_concepts.ipynb` §3 (Dot Products and Matrix
  Algebra) links here for the deep dive.
