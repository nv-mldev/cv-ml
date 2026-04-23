"""
Part 1 — Vectors and Dot Product
=================================
Demonstrates how image patches become vectors in high-dimensional space and
how the dot product measures pixel-by-pixel agreement between patches.

Run standalone:
    python part1_vectors_and_dot_product.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Consistent color palette for all plots
COLORS = {
    'primary': '#2196F3',     # blue
    'secondary': '#4CAF50',   # green
    'result': '#FFC107',      # amber
    'highlight': '#F44336',   # red
    'transform': '#9C27B0',   # purple
    'gradient': '#FF9800',    # orange
}


def make_gradient_patch(size: int = 3, start: float = 100, end: float = 200) -> np.ndarray:
    """Create a horizontal gradient patch (left=dark, right=bright)."""
    row = np.linspace(start, end, size)
    return np.tile(row, (size, 1))  # repeat the row vertically


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


print("Setup complete.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Create a 3×3 gradient patch (a tiny crop from a real image)
# 2. Flatten the 2D grid into a 1D vector of 9 pixel values
# 3. Plot the patch as an image and the vector as a bar chart
# What to look for: the same values appear in both; the patch is
#   just a reshaped view of the vector
# ───────────────────────────────────────────────────────────

patch = make_gradient_patch(size=3, start=50, end=200)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Show the 2D image
show_patch(patch, title='3×3 Image Patch (2D grid)', ax=axes[0])

# Show the flattened vector
vector = patch.flatten()
axes[1].barh(range(len(vector)), vector, color=COLORS['primary'], alpha=0.8)
axes[1].set_yticks(range(len(vector)))
axes[1].set_yticklabels([f'pixel[{i}]' for i in range(len(vector))], fontsize=9)
axes[1].set_xlabel('Pixel Value', fontsize=11)
axes[1].set_title('Flattened to 9D Vector', fontsize=12)
axes[1].invert_yaxis()
axes[1].set_xlim(0, 255)

plt.tight_layout()
plt.show()

print(f"2D patch shape: {patch.shape}")
print(f"Flattened vector: {vector}")
print(f"This vector is a point in {len(vector)}-dimensional space.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Create three 3-pixel patches with different patterns
# 2. Plot each as a point AND as an arrow from the origin in 3D space
# What to look for: patches with different patterns land at different
#   positions in pixel-value space — distance encodes dissimilarity
# ───────────────────────────────────────────────────────────

patch_a = np.array([50, 100, 200])     # dark-to-bright gradient
patch_b = np.array([200, 100, 50])     # bright-to-dark gradient (reversed)
patch_c = np.array([150, 150, 150])    # flat gray (no pattern)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot each patch as a point in 3D pixel-value space
for patch_vec, name, color in [(patch_a, 'Gradient →', COLORS['primary']),
                                (patch_b, 'Gradient ←', COLORS['highlight']),
                                (patch_c, 'Flat gray', COLORS['secondary'])]:
    ax.scatter(*patch_vec, s=120, color=color, label=f'{name} {patch_vec}', zorder=5)
    # Draw line from origin to the point (the vector)
    ax.plot([0, patch_vec[0]], [0, patch_vec[1]], [0, patch_vec[2]],
            color=color, linewidth=1.5, alpha=0.6)

ax.set_xlabel('Pixel 0', fontsize=11)
ax.set_ylabel('Pixel 1', fontsize=11)
ax.set_zlabel('Pixel 2', fontsize=11)
ax.set_title('Three Image Patches as Points in 3D Pixel Space', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()


# ── Algorithm ──────────────────────────────────────────────
# 1. Take a template patch and one similar, one opposite patch
# 2. Compute the raw dot product for each pair
# 3. Show element-wise products (the terms in the sum) as bar charts
# What to look for: the similar patch produces a higher dot product;
#   large aligned values contribute large positive terms to the sum
# ───────────────────────────────────────────────────────────

template = np.array([50, 150, 250], dtype=float)   # dark → bright
similar_patch = np.array([60, 140, 230], dtype=float)  # similar gradient
opposite_patch = np.array([250, 150, 50], dtype=float)  # reversed gradient

dot_similar = np.dot(template, similar_patch)
dot_opposite = np.dot(template, opposite_patch)

print("Template:        ", template.astype(int))
print("Similar patch:   ", similar_patch.astype(int))
print("Opposite patch:  ", opposite_patch.astype(int))
print()
print(f"template · similar  = {50}×{60} + {150}×{140} + {250}×{230} = {dot_similar:.0f}")
print(f"template · opposite = {50}×{250} + {150}×{150} + {250}×{50} = {dot_opposite:.0f}")
print()
print(f"Higher dot product ({dot_similar:.0f} > {dot_opposite:.0f}) = more agreement.")
print("But raw dot product also depends on magnitude — we'll fix that in Part 2.")

fig, axes = plt.subplots(2, 4, figsize=(16, 7))

for row, (patch, name) in enumerate([(similar_patch, 'Similar'),
                                      (opposite_patch, 'Opposite')]):
    # Show template
    axes[row, 0].bar(range(3), template, color=COLORS['primary'], alpha=0.8)
    axes[row, 0].set_title('Template' if row == 0 else '', fontsize=11)
    axes[row, 0].set_ylabel(name, fontsize=12, fontweight='bold')
    axes[row, 0].set_ylim(0, 280)
    axes[row, 0].set_xticks(range(3), ['p0', 'p1', 'p2'])

    # Show patch
    axes[row, 1].bar(range(3), patch, color=COLORS['secondary'], alpha=0.8)
    axes[row, 1].set_title('Patch' if row == 0 else '', fontsize=11)
    axes[row, 1].set_ylim(0, 280)
    axes[row, 1].set_xticks(range(3), ['p0', 'p1', 'p2'])

    # Element-wise product
    products = template * patch
    axes[row, 2].bar(range(3), products, color=COLORS['result'], alpha=0.8)
    axes[row, 2].set_title('Element-wise product' if row == 0 else '', fontsize=11)
    axes[row, 2].set_xticks(range(3), ['p0', 'p1', 'p2'])
    for i, val in enumerate(products):
        axes[row, 2].text(i, val + 500, f'{val:.0f}', ha='center', fontsize=9)

    # Sum = dot product
    dot = np.sum(products)
    axes[row, 3].bar([0], [dot], color=COLORS['gradient'], alpha=0.8, width=0.5)
    axes[row, 3].set_title('Dot product (sum)' if row == 0 else '', fontsize=11)
    axes[row, 3].text(0, dot + 1000, f'{dot:.0f}', ha='center', fontsize=12, fontweight='bold')
    axes[row, 3].set_xticks([])

plt.suptitle('Dot Product = Multiply Corresponding Pixels, Then Sum', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
