"""
Part 3 — Orthogonality and Projection
=======================================
Demonstrates orthogonality between spatial patterns and shows that mean
subtraction is exactly orthogonal projection onto the [1,1,...,1] brightness
direction — making brightness and pattern independent.

Run standalone:
    python part3_orthogonality_and_projection.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}


def make_gradient_patch(size: int = 3, start: float = 100, end: float = 200) -> np.ndarray:
    """Create a horizontal gradient patch (left=dark, right=bright)."""
    row = np.linspace(start, end, size)
    return np.tile(row, (size, 1))


def show_patch(patch: np.ndarray, title: str = "", ax=None, show_values: bool = True):
    """Display a small image patch with pixel values overlaid."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(patch, cmap='gray', vmin=0, vmax=255)
    if show_values:
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                ax.text(j, i, f'{patch[i, j]:.0f}', ha='center', va='center',
                        color='red' if patch[i, j] > 127 else 'yellow', fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])


print("Setup complete.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Create horizontal and vertical stripe patterns (2×2 patches)
# 2. Compute their dot product
# 3. Verify they become orthogonal after mean-centering
# What to look for: raw dot product is nonzero; after centering it is zero,
#   proving these two patterns are orthogonal directions in image space
# ───────────────────────────────────────────────────────────

horizontal_pattern = np.array([0, 255, 0, 255], dtype=float)  # columns differ
vertical_pattern = np.array([0, 0, 255, 255], dtype=float)    # rows differ

dot = np.dot(horizontal_pattern, vertical_pattern)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
show_patch(horizontal_pattern.reshape(2, 2), 'Horizontal stripes', ax=axes[0])
show_patch(vertical_pattern.reshape(2, 2), 'Vertical stripes', ax=axes[1])

plt.suptitle(f'Dot product = {dot:.0f} → Orthogonal!', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"Horizontal: {horizontal_pattern.astype(int)}")
print(f"Vertical:   {vertical_pattern.astype(int)}")
print(f"Dot product: (0)(0) + (255)(0) + (0)(255) + (255)(255) = {dot:.0f}")
print()

# Verify centered versions are orthogonal
h_centered = horizontal_pattern - np.mean(horizontal_pattern)
v_centered = vertical_pattern - np.mean(vertical_pattern)
dot_centered = np.dot(h_centered, v_centered)

print(f"Centered horizontal: {h_centered}")
print(f"Centered vertical:   {v_centered}")
print(f"Dot product of centered: {dot_centered:.0f} → Orthogonal after centering! ✓")
print("\nThese patterns capture independent spatial information — orthogonal basis patterns.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Show that scalar multiples of [1,1,...,1] produce flat gray images
# 2. Visualize four uniform patches at increasing brightness levels
# What to look for: each patch is featureless — pure brightness, zero pattern
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

for ax, brightness, label in [(axes[0], 50, '50·[1,1,...,1]'),
                                (axes[1], 128, '128·[1,1,...,1]'),
                                (axes[2], 200, '200·[1,1,...,1]'),
                                (axes[3], 255, '255·[1,1,...,1]')]:
    uniform_patch = np.full((4, 4), brightness, dtype=float)
    show_patch(uniform_patch, f'{label}\n= uniform gray', ax=ax, show_values=False)

plt.suptitle('The [1,1,...,1] Direction = Pure Brightness, Zero Pattern', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

print("Any scalar multiple of [1,1,...,1] is a flat gray image.")
print("This direction encodes ONLY brightness — no spatial pattern at all.")
print("Mean subtraction removes exactly this component.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Take a gradient patch (has both brightness and pattern)
# 2. Compute brightness component = mean × [1,1,...,1]
# 3. Compute pattern component = patch − brightness
# 4. Display the decomposition as a three-panel figure
# 5. Verify brightness · pattern = 0 (orthogonal)
# What to look for: pattern sums to zero; dot product is zero
# ───────────────────────────────────────────────────────────

gradient = make_gradient_patch(size=4, start=80, end=220)

mean_value = np.mean(gradient)
brightness_component = np.full_like(gradient, mean_value)
pattern_component = gradient - brightness_component

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

show_patch(gradient, f'Original\nmean={mean_value:.0f}', ax=axes[0])

axes[1].text(0.5, 0.5, '=', fontsize=40, ha='center', va='center', transform=axes[1].transAxes)
axes[1].axis('off')

show_patch(brightness_component, f'Brightness\n(mean·[1,1,...,1])', ax=axes[2])

axes[3].text(0.5, 0.5, '+', fontsize=40, ha='center', va='center', transform=axes[3].transAxes)
axes[3].axis('off')

axes[4].imshow(pattern_component, cmap='RdBu_r', vmin=-100, vmax=100)
for i in range(pattern_component.shape[0]):
    for j in range(pattern_component.shape[1]):
        axes[4].text(j, i, f'{pattern_component[i, j]:.0f}', ha='center', va='center',
                     fontsize=9, color='black')
axes[4].set_title(f'Pattern\n(residual, sums to {np.sum(pattern_component):.0f})', fontsize=12)
axes[4].set_xticks([])
axes[4].set_yticks([])

plt.suptitle('Orthogonal Decomposition: Image = Brightness + Pattern', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()

dot_product = np.dot(brightness_component.flatten(), pattern_component.flatten())
print(f"Dot product of brightness and pattern components: {dot_product:.6f}")
print(f"Orthogonal? {np.isclose(dot_product, 0)} ← These components are independent!")


# ── Algorithm ──────────────────────────────────────────────
# 1. Create three versions of the same patch at different brightness offsets
# 2. For each, extract the brightness and pattern components
# 3. Show in a 3×3 grid: original | brightness | pattern
# 4. Confirm pattern arrays are identical across all three rows
# What to look for: the rightmost column (pattern) is the same for all rows
# ───────────────────────────────────────────────────────────

original = make_gradient_patch(size=4, start=80, end=220)
brighter = original + 60
darker = original - 40

fig, axes = plt.subplots(3, 3, figsize=(14, 12))

for row, (patch, label) in enumerate([(original, 'Original'),
                                       (brighter, 'Brighter (+60)'),
                                       (darker, 'Darker (-40)')]):
    mean_val = np.mean(patch)
    brightness = np.full_like(patch, mean_val)
    pattern = patch - brightness

    show_patch(np.clip(patch, 0, 255), f'{label}\nmean={mean_val:.0f}', ax=axes[row, 0], show_values=False)
    show_patch(np.clip(brightness, 0, 255), f'Brightness = {mean_val:.0f}', ax=axes[row, 1], show_values=False)

    axes[row, 2].imshow(pattern, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[row, 2].set_title(f'Pattern (SAME!)', fontsize=12, color=COLORS['secondary'], fontweight='bold')
    axes[row, 2].set_xticks([])
    axes[row, 2].set_yticks([])

axes[0, 0].set_ylabel('Original', fontsize=12)
axes[1, 0].set_ylabel('Brighter +60', fontsize=12)
axes[2, 0].set_ylabel('Darker -40', fontsize=12)

plt.suptitle('Mean Subtraction: Different Brightness → Same Pattern Component', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

patterns = [p - np.mean(p) for p in [original, brighter, darker]]
print(f"Pattern from original == pattern from brighter? {np.allclose(patterns[0], patterns[1])}")
print(f"Pattern from original == pattern from darker?   {np.allclose(patterns[0], patterns[2])}")
print("\n→ Mean subtraction removes uniform brightness. Pattern is preserved.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Manually work through the projection formula step by step
# 2. Show that (v · û)û equals mean × [1,1,...,1]
# 3. Verify the residual is orthogonal to [1,1,...,1]
# What to look for: every intermediate quantity confirms the same result
# ───────────────────────────────────────────────────────────

vector = np.array([80, 120, 180, 220], dtype=float)
n = len(vector)

# Step 1: unit vector along [1,1,1,1]
ones = np.ones(n)
u_hat = ones / np.sqrt(n)
print(f"û = [1,1,1,1] / √4 = {u_hat}")

# Step 2: dot product v · û
dot = np.dot(vector, u_hat)
print(f"v · û = {dot:.2f}")

# Step 3: projection = (v · û) * û
projection = dot * u_hat
print(f"Projection = {dot:.2f} × {u_hat} = {projection}")

# Step 4: this equals mean · [1,1,1,1]
mean_val = np.mean(vector)
mean_broadcast = mean_val * ones
print(f"Mean × [1,1,1,1] = {mean_val} × [1,1,1,1] = {mean_broadcast}")
print(f"Equal? {np.allclose(projection, mean_broadcast)}")

# Step 5: residual = v - projection = mean-subtracted vector
residual = vector - projection
print(f"\nResidual = v - projection = {residual}")
print(f"v - mean = {vector - mean_val}")
print(f"Equal? {np.allclose(residual, vector - mean_val)}")
print(f"\nDot product of residual with [1,1,1,1]: {np.dot(residual, ones):.10f} ← zero = orthogonal ✓")
