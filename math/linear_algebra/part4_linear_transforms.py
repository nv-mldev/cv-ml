"""
Part 4 — Linear Transforms
============================
Demonstrates the lighting model I = aT + b as a general linear transform, its
geometric effect in pixel space, and the difference between general and
orthogonal (energy-preserving) transforms including the DFT.

Run standalone:
    python part4_linear_transforms.py
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


print("Setup complete.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Apply six (a, b) combinations to a checkerboard template
# 2. Show each result as a grayscale image
# What to look for: all six images show the same checkerboard — only
#   brightness and contrast vary, not the spatial pattern
# ───────────────────────────────────────────────────────────

template_img = make_checkerboard(size=6, low=60, high=200)

transforms = [
    (1.0, 0,   'Original (a=1, b=0)'),
    (1.0, 50,  'Brightness +50 (a=1, b=50)'),
    (1.5, 0,   'High contrast (a=1.5, b=0)'),
    (0.5, 0,   'Low contrast (a=0.5, b=0)'),
    (1.5, 50,  'Both (a=1.5, b=50)'),
    (0.5, 128, 'Washed out (a=0.5, b=128)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, (a, b, title) in zip(axes.flatten(), transforms):
    transformed = np.clip(a * template_img + b, 0, 255)
    show_patch(transformed, title, ax=ax, show_values=False)

plt.suptitle('Linear Transforms of a Checkerboard: I = a·T + b', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("All six images have the SAME checkerboard pattern.")
print("They differ only in brightness (b) and contrast (a).")
print("A good matching method should recognize all of them as the same pattern.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Use 3-pixel vectors to visualize in 3D
# 2. Left plot: scale a by 0.5, 1.0, 2.0 → all arrows in the same direction
# 3. Right plot: add b = 0, 50, 100 → arrows drift toward the [1,1,1] diagonal
# What to look for: scaling keeps the direction constant; offset changes it
# ───────────────────────────────────────────────────────────

template_vec = np.array([60, 130, 200], dtype=float)

fig = plt.figure(figsize=(14, 6))

# Left: scaling (a) — stretches along the same direction
ax1 = fig.add_subplot(121, projection='3d')
origin = [0, 0, 0]

for a, color, label in [(0.5, COLORS['secondary'], 'a=0.5 (dim)'),
                         (1.0, COLORS['primary'], 'a=1.0 (original)'),
                         (2.0, COLORS['highlight'], 'a=2.0 (bright)')]:
    scaled = a * template_vec
    scale_factor = 1.0 / 500
    ax1.quiver(*origin, *(scaled * scale_factor), color=color,
               arrow_length_ratio=0.08, linewidth=2, label=label)

ax1.set_title('Scaling (a): Same Direction,\nDifferent Length', fontsize=12)
ax1.set_xlabel('P0')
ax1.set_ylabel('P1')
ax1.set_zlabel('P2')
ax1.legend(fontsize=9)

# Right: offset (b) — shifts toward [1,1,1] diagonal
ax2 = fig.add_subplot(122, projection='3d')

for b, color, label in [(0, COLORS['primary'], 'b=0 (original)'),
                         (50, COLORS['gradient'], 'b=50'),
                         (100, COLORS['highlight'], 'b=100')]:
    offset_vec = template_vec + b
    scale_factor = 1.0 / 500
    ax2.quiver(*origin, *(offset_vec * scale_factor), color=color,
               arrow_length_ratio=0.08, linewidth=2, label=label)

# Show the [1,1,1] direction
diag = np.ones(3) / np.linalg.norm(np.ones(3)) * 0.8
ax2.plot([0, diag[0]], [0, diag[1]], [0, diag[2]], '--', color='gray', alpha=0.5, label='[1,1,1]')

ax2.set_title('Offset (b): Direction Changes\n(shifts toward [1,1,1])', fontsize=12)
ax2.set_xlabel('P0')
ax2.set_ylabel('P1')
ax2.set_zlabel('P2')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.show()

print("Scaling (a): changes LENGTH but not DIRECTION → cosine similarity handles this")
print("Offset (b): changes DIRECTION (toward [1,1,1]) → cosine similarity FAILS")
print("→ Need mean subtraction to remove the [1,1,1] component before comparing directions")


# ── Algorithm ──────────────────────────────────────────────
# 1. Apply a random (non-orthogonal) matrix to a 3-element signal
# 2. Compare original and transformed energies
# What to look for: energy changes — the matrix stretches or compresses the signal
# ───────────────────────────────────────────────────────────

signal = np.array([100, 150, 200], dtype=float)
original_energy = np.linalg.norm(signal) ** 2

np.random.seed(42)
A = np.random.randn(3, 3)

transformed = A @ signal
transformed_energy = np.linalg.norm(transformed) ** 2

print("=== General Linear Transform (non-orthogonal matrix) ===")
print(f"Original signal:     {signal.astype(int)}")
print(f"Original energy:     {original_energy:.1f}")
print(f"Transformed signal:  {transformed.round(1)}")
print(f"Transformed energy:  {transformed_energy:.1f}")
print(f"Energy ratio:        {transformed_energy / original_energy:.3f}")
print(f"Energy preserved?    {np.isclose(original_energy, transformed_energy)}  ← NO!")
print()
print("A general linear transform changes the energy of the signal.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Apply a rotation matrix (orthogonal) to the same signal
# 2. Verify Q^T Q = I and energy is preserved
# What to look for: rotated energy equals original energy exactly
# ───────────────────────────────────────────────────────────

theta = np.pi / 4  # 45-degree rotation
Q_rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,              0,             1]
])

rotated = Q_rotation @ signal
rotated_energy = np.linalg.norm(rotated) ** 2

print("=== Orthogonal Transform (rotation matrix) ===")
print(f"Original signal:   {signal.astype(int)}")
print(f"Original energy:   {original_energy:.1f}")
print(f"Rotated signal:    {rotated.round(1)}")
print(f"Rotated energy:    {rotated_energy:.1f}")
print(f"Energy preserved?  {np.isclose(original_energy, rotated_energy)}  ← YES!")
print()
print(f"Q^T Q = I?  {np.allclose(Q_rotation.T @ Q_rotation, np.eye(3))}  ← Orthogonal matrix ✓")


# ── Algorithm ──────────────────────────────────────────────
# 1. Build a synthetic 1D signal with two frequency components
# 2. Compute its DFT using np.fft.fft
# 3. Compute energy in both domains and compare (Parseval's theorem)
# 4. Plot the signal and its magnitude spectrum side by side
# What to look for: two peaks in the spectrum at frequencies 3 and 7;
#   spatial energy equals frequency energy (Parseval's holds)
# ───────────────────────────────────────────────────────────

n_pixels = 64
x = np.arange(n_pixels, dtype=float)
signal_1d = 100 + 50 * np.sin(2 * np.pi * 3 * x / n_pixels) + 30 * np.cos(2 * np.pi * 7 * x / n_pixels)

spectrum = np.fft.fft(signal_1d)

spatial_energy = np.sum(np.abs(signal_1d) ** 2)
frequency_energy = np.sum(np.abs(spectrum) ** 2) / n_pixels  # Parseval's: divide by n for DFT convention

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(signal_1d, color=COLORS['primary'], linewidth=2)
axes[0].fill_between(range(n_pixels), signal_1d, alpha=0.2, color=COLORS['primary'])
axes[0].set_title(f'Spatial Domain\nEnergy = Σ|x|² = {spatial_energy:.0f}', fontsize=12)
axes[0].set_xlabel('Pixel Position', fontsize=11)
axes[0].set_ylabel('Pixel Value', fontsize=11)
axes[0].grid(True, alpha=0.3)

freqs = np.arange(n_pixels // 2)
magnitudes = np.abs(spectrum[:n_pixels // 2])
axes[1].stem(freqs, magnitudes, linefmt=COLORS['highlight'], markerfmt='o', basefmt='gray')
axes[1].set_title(f'Frequency Domain\nEnergy = (1/n)·Σ|X|² = {frequency_energy:.0f}', fontsize=12)
axes[1].set_xlabel('Frequency Bin', fontsize=11)
axes[1].set_ylabel('Magnitude |X[k]|', fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.suptitle(
    f"Parseval's Theorem: Energy Preserved Across Domains ({spatial_energy:.0f} ≈ {frequency_energy:.0f})",
    fontsize=14, y=1.02
)
plt.tight_layout()
plt.show()

print(f"Spatial energy:   {spatial_energy:.2f}")
print(f"Frequency energy: {frequency_energy:.2f}")
print(f"Equal? {np.isclose(spatial_energy, frequency_energy)}")
print()
print("The DFT just ROTATES the signal into a different basis (frequency instead of position).")
print("No energy is created or destroyed — it is redistributed across frequency bins.")


# ── Algorithm ──────────────────────────────────────────────
# 1. Build a comparison table of six transforms
# 2. For each, compute output energy and ratio to original energy
# 3. Mark which transforms preserve energy
# What to look for: only rotation and DFT preserve energy; all others change it
# ───────────────────────────────────────────────────────────

signal = np.array([100, 150, 200], dtype=float)
original_energy = np.linalg.norm(signal) ** 2

print(f"Original signal: {signal.astype(int)},  energy = {original_energy:.0f}")
print("─" * 70)
print(f"{'Transform':35s}  {'Energy':>10s}  {'Ratio':>8s}  {'Preserved?':>10s}")
print("─" * 70)

test_transforms = [
    ("Contrast ×2 (I = 2T)",         2 * signal),
    ("Contrast ×0.5 (I = 0.5T)",     0.5 * signal),
    ("Brightness +50 (I = T + 50)",   signal + 50),
    ("Both (I = 2T + 50)",            2 * signal + 50),
    ("Rotation (orthogonal)",          Q_rotation @ signal),
    ("DFT (orthogonal)",              np.fft.fft(signal)),
]

for name, result in test_transforms:
    energy = np.sum(np.abs(result) ** 2)
    # For DFT, divide by n for Parseval's convention
    if 'DFT' in name:
        energy = energy / len(signal)
    ratio = energy / original_energy
    preserved = np.isclose(energy, original_energy)
    print(f"{name:35s}  {energy:>10.0f}  {ratio:>8.3f}  {'✓ YES' if preserved else '✗ NO':>10s}")

print("─" * 70)
print()
print("Key insight:")
print("• General linear transforms (scaling, offset, both) CHANGE energy")
print("• Orthogonal transforms (rotation, DFT, DCT) PRESERVE energy")
print("• Orthogonal = pure rotation of the coordinate system, no stretching")
print("• Fourier IS linear AND orthogonal — that is why it preserves energy")
