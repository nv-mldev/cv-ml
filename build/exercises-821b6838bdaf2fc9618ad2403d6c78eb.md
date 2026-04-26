# Exercises

Three exercises to consolidate the concepts from Parts 1–4.

---

## Exercise 1: Flatten and Compare

**Task:** Create two 4×4 synthetic images — a horizontal gradient and a
vertical gradient. Flatten both to vectors. Compute their dot product and
cosine similarity. Are they orthogonal?

**Hint:** A horizontal gradient has the same value in each column. A vertical
gradient has the same value in each row.

**Expected output:** The dot product should be nonzero (they are not orthogonal
in general), but after centering, they should become orthogonal.

---

## Exercise 2: Energy Change Under Transforms

**Task:** Take a 3×3 checkerboard patch. Apply these transforms and compute
the energy (squared L2 norm) of each:

1. Multiply by 3 (contrast increase)
2. Add 100 (brightness increase)
3. Rotate the flattened vector using a random orthogonal matrix (use
   `np.linalg.qr` to generate one)

Which transforms preserve energy?

**Hint:** `Q, R = np.linalg.qr(np.random.randn(9, 9))` gives you a 9×9
orthogonal matrix Q.

**Expected output:** Only the rotation preserves energy.

---

## Exercise 3: Decompose a Real-ish Image

**Task:** Create an 8×8 synthetic image that has both a pattern (a diagonal
stripe) and brightness (mean around 150). Decompose it into brightness +
pattern components. Verify orthogonality. Then add a brightness offset of +80
and show the pattern component is unchanged.

**Hint:** A diagonal stripe: `image[i, j] = 255 if abs(i - j) < 2 else 50`

**Expected output:** Same pattern component before and after brightness offset.
