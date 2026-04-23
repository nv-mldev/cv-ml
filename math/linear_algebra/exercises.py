"""
Exercises — Linear Algebra for Matching
=========================================
Three exercises to consolidate the concepts from Parts 1–4.

Fill in each function where marked # YOUR CODE HERE.
Run standalone:
    python exercises.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Exercise 1: Flatten and Compare
# ---------------------------------------------------------------------------

def exercise_1_flatten_and_compare() -> None:
    """
    Create a 4×4 horizontal gradient and a 4×4 vertical gradient.
    Flatten both to vectors. Compute their dot product and cosine similarity.
    Then center both vectors and repeat.

    Expected output:
        - Raw dot product is nonzero.
        - After centering, dot product is zero (orthogonal).
    """
    # YOUR CODE HERE
    # Step 1: create a 4×4 horizontal gradient
    #   (same value across each row, values increase column by column)
    horizontal_gradient = None  # replace with np.ndarray

    # Step 2: create a 4×4 vertical gradient
    #   (same value across each column, values increase row by row)
    vertical_gradient = None  # replace with np.ndarray

    # Step 3: flatten both patches to 1D vectors
    h_flat = None  # horizontal_gradient.flatten()
    v_flat = None  # vertical_gradient.flatten()

    # Step 4: compute dot product of raw (un-centered) vectors
    raw_dot = None  # np.dot(h_flat, v_flat)

    # Step 5: compute cosine similarity of raw vectors
    raw_cos = None  # np.dot(h_flat, v_flat) / (np.linalg.norm(h_flat) * np.linalg.norm(v_flat))

    # Step 6: center both vectors (subtract their means)
    h_centered = None  # h_flat - np.mean(h_flat)
    v_centered = None  # v_flat - np.mean(v_flat)

    # Step 7: compute dot product of centered vectors
    centered_dot = None  # np.dot(h_centered, v_centered)

    print("=== Exercise 1: Flatten and Compare ===")
    print(f"Raw dot product:      {raw_dot}")
    print(f"Raw cosine similarity: {raw_cos:.4f}" if raw_cos is not None else "Raw cosine: (not computed)")
    print(f"Centered dot product: {centered_dot}")
    print(f"Orthogonal after centering? {np.isclose(centered_dot, 0) if centered_dot is not None else 'unknown'}")
    print()


# ---------------------------------------------------------------------------
# Exercise 2: Energy Change Under Transforms
# ---------------------------------------------------------------------------

def exercise_2_energy_under_transforms() -> None:
    """
    Create a 3×3 checkerboard patch. Apply contrast ×3, brightness +100,
    and a random orthogonal rotation. Compare the energy of each result.

    Expected output:
        - Contrast ×3: energy increases by ×9
        - Brightness +100: energy increases
        - Rotation: energy unchanged
    """
    # YOUR CODE HERE
    # Step 1: create a 3×3 checkerboard (use your own values or a helper)
    checkerboard = None  # np.ndarray of shape (3, 3)

    # Step 2: flatten to a 9-element vector
    flat = None  # checkerboard.flatten()

    # Step 3: compute original energy = squared L2 norm
    original_energy = None  # np.sum(flat ** 2)

    # Step 4: apply contrast ×3 and compute energy
    contrast_result = None  # 3 * flat
    contrast_energy = None  # np.sum(contrast_result ** 2)

    # Step 5: apply brightness +100 and compute energy
    brightness_result = None  # flat + 100
    brightness_energy = None  # np.sum(brightness_result ** 2)

    # Step 6: generate a random 9×9 orthogonal matrix using QR decomposition
    np.random.seed(0)
    Q = None  # Q, _ = np.linalg.qr(np.random.randn(9, 9)); use Q

    # Step 7: apply the rotation and compute energy
    rotated_result = None  # Q @ flat
    rotated_energy = None  # np.sum(rotated_result ** 2)

    print("=== Exercise 2: Energy Change Under Transforms ===")
    print(f"Original energy:    {original_energy}")
    print(f"Contrast ×3 energy: {contrast_energy}")
    print(f"Brightness +100:    {brightness_energy}")
    print(f"Rotation energy:    {rotated_energy}")
    print(f"Rotation preserves energy? {np.isclose(original_energy, rotated_energy) if (original_energy is not None and rotated_energy is not None) else 'unknown'}")
    print()


# ---------------------------------------------------------------------------
# Exercise 3: Decompose a Real-ish Image
# ---------------------------------------------------------------------------

def exercise_3_decompose_image() -> None:
    """
    Create an 8×8 diagonal-stripe image. Decompose it into brightness +
    pattern components and verify orthogonality. Then add a brightness
    offset of +80 and show the pattern component is unchanged.

    Expected output:
        - Dot product of brightness and pattern components ≈ 0
        - Pattern arrays are identical before and after brightness offset
    """
    # YOUR CODE HERE
    # Step 1: build the 8×8 diagonal stripe image
    #   pixel value = 255 if abs(i - j) < 2 else 50
    image = None  # np.ndarray of shape (8, 8)

    # Step 2: compute brightness component = mean broadcast to all pixels
    mean_val = None  # np.mean(image)
    brightness_component = None  # np.full_like(image, mean_val, dtype=float)

    # Step 3: compute pattern component = image - brightness
    pattern_component = None  # image - brightness_component

    # Step 4: verify orthogonality (dot product should be ≈ 0)
    dot_ortho = None  # np.dot(brightness_component.flatten(), pattern_component.flatten())

    # Step 5: create a brighter version of the image (add +80)
    image_brighter = None  # image + 80

    # Step 6: extract the pattern component from the brighter image
    pattern_brighter = None  # image_brighter - np.mean(image_brighter)

    # Step 7: verify the pattern is unchanged
    patterns_equal = None  # np.allclose(pattern_component, pattern_brighter)

    print("=== Exercise 3: Decompose a Real-ish Image ===")
    print(f"Dot product (brightness · pattern): {dot_ortho}")
    print(f"Orthogonal? {np.isclose(dot_ortho, 0) if dot_ortho is not None else 'unknown'}")
    print(f"Pattern unchanged after +80 offset? {patterns_equal}")
    print()


# ---------------------------------------------------------------------------
# Run all exercises
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    exercise_1_flatten_and_compare()
    exercise_2_energy_under_transforms()
    exercise_3_decompose_image()
