"""
Part 2 — Norms and Similarity
==============================
Demonstrates the L2 norm as image energy, unit vector normalisation, and
cosine similarity — including the brightness-offset blind spot.

Run standalone:
    python part2_norms_and_similarity.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}


def make_checkerboard(size: int = 4, low: float = 50, high: float = 200) -> np.ndarray:
    """Create a checkerboard pattern."""
    board = np.zeros((size, size))
    board[0::2, 0::2] = high
    board[1::2, 1::2] = high
    board[0::2, 1::2] = low
    board[1::2, 0::2] = low
    return board


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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print("Setup complete.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Create the same checkerboard pattern at three brightness levels
# 2. Compute the L2 norm for each
# 3. Display patches side by side with their norms in the title
# What to look for: same pattern, very different norms — the norm
#   captures brightness energy, not the spatial pattern
# ───────────────────────────────────────────────────────────

dark_checker = make_checkerboard(size=4, low=20, high=80)
medium_checker = make_checkerboard(size=4, low=50, high=200)
bright_checker = make_checkerboard(size=4, low=100, high=255)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, patch, label in [(axes[0], dark_checker, 'Dark'),
                          (axes[1], medium_checker, 'Medium'),
                          (axes[2], bright_checker, 'Bright')]:
    l2_norm = np.linalg.norm(patch.flatten())
    show_patch(patch, title=f'{label}\nL2 norm = {l2_norm:.1f}', ax=ax, show_values=False)

plt.suptitle('Same Pattern, Different Brightness → Different L2 Norms', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"Dark checkerboard  L2 norm: {np.linalg.norm(dark_checker.flatten()):.1f}")
print(f"Medium checkerboard L2 norm: {np.linalg.norm(medium_checker.flatten()):.1f}")
print(f"Bright checkerboard L2 norm: {np.linalg.norm(bright_checker.flatten()):.1f}")
print("\nL2 norm captures brightness (energy), NOT pattern.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Show the Pythagorean-theorem extension to n dimensions
# 2. Show that ‖v‖² = v · v (dot product with itself)
# What to look for: both computations give identical results
# ───────────────────────────────────────────────────────────

vector = np.array([100, 150, 200], dtype=float)
l2_norm = np.sqrt(100**2 + 150**2 + 200**2)

print(f"Vector: {vector.astype(int)}")
print(f"L2 norm = √(100² + 150² + 200²) = √{100**2 + 150**2 + 200**2} = {l2_norm:.1f}")
print(f"np.linalg.norm: {np.linalg.norm(vector):.1f}")
print()
print("The L2 norm is the same as the Pythagorean theorem extended to n dimensions.")

l2_squared = np.linalg.norm(vector) ** 2
dot_with_self = np.dot(vector, vector)

print(f"‖v‖² = {l2_squared:.1f}")
print(f"v · v = {dot_with_self:.1f}")
print(f"Equal? {np.isclose(l2_squared, dot_with_self)}")
print()
print("This is why the squared L2 norm is called the 'energy' of the signal.")
print(f"Energy of {vector.astype(int)} = {dot_with_self:.0f}")


# ── Algorithm ──────────────────────────────────────────────
# 1. Take three versions of the same gradient at different brightness
# 2. Divide each by its L2 norm to get unit vectors
# 3. Verify all unit vectors are identical
# What to look for: same direction (unit vector) despite different lengths
# ───────────────────────────────────────────────────────────

dim_patch = np.array([50, 100, 150], dtype=float)
medium_patch = np.array([100, 200, 300], dtype=float)   # 2× brighter
bright_patch_vec = np.array([150, 300, 450], dtype=float)   # 3× brighter

dim_unit = dim_patch / np.linalg.norm(dim_patch)
medium_unit = medium_patch / np.linalg.norm(medium_patch)
bright_unit = bright_patch_vec / np.linalg.norm(bright_patch_vec)

print("Patch              Vector                  Unit Vector                 Length")
print("─" * 85)
for name, vec, unit in [('Dim (1×)', dim_patch, dim_unit),
                         ('Medium (2×)', medium_patch, medium_unit),
                         ('Bright (3×)', bright_patch_vec, bright_unit)]:
    print(f"{name:14s}  {str(vec.astype(int)):22s}  →  {np.array2string(unit, precision=4):30s}  ‖v‖={np.linalg.norm(vec):.1f}")

print()
print(f"All unit vectors identical? {np.allclose(dim_unit, medium_unit) and np.allclose(medium_unit, bright_unit)}")
print("Brightness removed. Pattern preserved.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Construct checker patches with same pattern, different brightness
# 2. Plot them on the unit sphere in 3D pixel space
# 3. Show that same-pattern patches collapse to the same sphere point
# What to look for: dark and bright checker arrows point the same way;
#   inverted pattern points a different direction
# ───────────────────────────────────────────────────────────

dark_row = np.array([30, 180, 30], dtype=float)
bright_row = np.array([90, 540, 90], dtype=float)    # 3× brighter (same pattern)
different_row = np.array([180, 30, 180], dtype=float) # inverted pattern

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw unit sphere (first octant)
phi = np.linspace(0, np.pi / 2, 30)
theta = np.linspace(0, np.pi / 2, 30)
phi_g, theta_g = np.meshgrid(phi, theta)
sx = np.sin(phi_g) * np.cos(theta_g)
sy = np.sin(phi_g) * np.sin(theta_g)
sz = np.cos(phi_g)
ax.plot_wireframe(sx, sy, sz, alpha=0.08, color='gray')

max_norm = max(np.linalg.norm(dark_row), np.linalg.norm(bright_row), np.linalg.norm(different_row))
scale = 1.0 / max_norm
origin = [0, 0, 0]

for vec, name, color in [(dark_row, 'Dark checker', COLORS['primary']),
                          (bright_row, 'Bright checker (3×)', COLORS['gradient']),
                          (different_row, 'Inverted pattern', COLORS['highlight'])]:
    ax.quiver(*origin, *(vec * scale), color=color, arrow_length_ratio=0.06,
              linewidth=2, label=f'{name} {vec.astype(int)}')
    unit = vec / np.linalg.norm(vec)
    ax.scatter(*unit, s=80, color=color, zorder=5)

ax.set_xlabel('Pixel 0', fontsize=11)
ax.set_ylabel('Pixel 1', fontsize=11)
ax.set_zlabel('Pixel 2', fontsize=11)
ax.set_title('Normalisation: Same Pattern → Same Point on Unit Sphere', fontsize=13)
ax.legend(loc='upper left', fontsize=9)
plt.tight_layout()
plt.show()

print("Dark and Bright checker point in the SAME direction → same point on sphere.")
print("Inverted pattern points in a DIFFERENT direction → different point on sphere.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Define cosine_similarity helper
# 2. Compare template against: same-pattern variants, flat gray, inverted
# 3. Print cos values and angles
# What to look for: same-pattern patches all give cos ≈ 1 regardless of brightness
# ───────────────────────────────────────────────────────────

template = np.array([50, 150, 250], dtype=float)

test_patches = [
    ("Same pattern, dim",      np.array([25, 75, 125], dtype=float)),
    ("Same pattern, bright",   np.array([100, 300, 500], dtype=float)),
    ("Slightly different",     np.array([60, 140, 230], dtype=float)),
    ("Flat gray (no pattern)", np.array([150, 150, 150], dtype=float)),
    ("Inverted gradient",      np.array([250, 150, 50], dtype=float)),
]

print(f"Template: {template.astype(int)}")
print("─" * 60)
for name, patch in test_patches:
    cos = cosine_similarity(template, patch)
    angle_deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    print(f"{name:25s}  {str(patch.astype(int)):18s}  cos={cos:+.4f}  angle={angle_deg:.1f}°")

print()
print("Same pattern at ANY brightness → cos ≈ 1.0 (angle ≈ 0°)")
print("This is exactly what TM_SQDIFF_NORMED uses — but it has a blind spot (Part 3).")


# ── Algorithm ──────────────────────────────────────────────
# 1. Compare template with a scaled version (should give cos ≈ 1)
# 2. Compare template with an offset version (cos will drop)
# 3. Bar chart showing the three vectors
# What to look for: scaling preserves direction; offset changes direction
# ───────────────────────────────────────────────────────────

template = np.array([50, 150, 250], dtype=float)
scaled = template * 3                     # same direction (scaling)
offset = template + 100                   # different direction (offset)

cos_scaled = cosine_similarity(template, scaled)
cos_offset = cosine_similarity(template, offset)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, patch, name, cos_val in [(axes[0], template, 'Template', 1.0),
                                   (axes[1], scaled, f'Scaled ×3 (cos={cos_scaled:.4f})', cos_scaled),
                                   (axes[2], offset, f'Offset +100 (cos={cos_offset:.4f})', cos_offset)]:
    ax.bar(range(3), patch, color=COLORS['primary'] if cos_val > 0.999 else COLORS['highlight'], alpha=0.8)
    ax.set_title(name, fontsize=11)
    ax.set_xticks(range(3), ['p0', 'p1', 'p2'])
    ax.set_ylim(0, 800)
    for i, val in enumerate(patch):
        ax.text(i, val + 10, f'{val:.0f}', ha='center', fontsize=10)

plt.suptitle('Cosine Similarity: Handles Scaling ✓, Fails on Offset ✗', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print(f"Scaled ×3:   cos = {cos_scaled:.4f}  → SAME direction (scaling preserved) ✓")
print(f"Offset +100: cos = {cos_offset:.4f}  → DIFFERENT direction (offset changed it) ✗")
print("\nThis is why TM_SQDIFF_NORMED fails on brightness offsets.")
print("We need mean subtraction (Part 3) to fix this.")
