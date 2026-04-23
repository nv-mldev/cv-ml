"""
Introduction to Probability for Computer Vision

Builds the complete conceptual ladder from random process to probability
distribution using a single running example — a 3-letter word source —
then grounds it in real image data.

What this script demonstrates:
  1. Random process → sample space → event → random variable → distribution
  2. The word source: vowel count as a Binomial random variable
  3. Two patches with identical means but different distributions (real image)
  4. Patch histograms as approximations of the true distribution
  5. Sampling from a distribution to generate new patches
  6. How distribution shape encodes physical image content
  7. Histogram convergence: more samples → closer to the true PMF/PDF
  8. Parameter compression: one number reconstructs an entire distribution

Run: python part0_what_is_a_distribution.py
"""

# --- Setup ---
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.stats import poisson, binom

COLORS = {
    'primary':   '#2196F3',
    'secondary': '#4CAF50',
    'result':    '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient':  '#FF9800',
}

np.random.seed(42)
print("Setup complete.")

# ── Algorithm ──────────────────────────────────────────────────────────────
# Step 1 — The Random Process and Sample Space
# 1. Define the word source: 26^3 = 17,576 equally likely 3-letter words
# 2. Run 10,000 trials — press the button 10,000 times
# 3. For each word, apply three different random variables:
#    X1 = number of vowels  (values: 0–3)
#    X2 = position of first letter  (values: 1–26)
#    X3 = is it a real English word?  (values: 0 or 1)
# 4. Plot the distribution of each random variable
# What to look for: same process, same word, three completely different
#   distributions depending on the question (random variable) you ask.
# ──────────────────────────────────────────────────────────────────────────

VOWELS = set('aeiou')
N_TRIALS = 10_000

# Simulate the word source: draw 3 random letters per trial
words = np.random.randint(0, 26, size=(N_TRIALS, 3))  # 0=a, 1=b, ..., 25=z

# Random variable X1: number of vowels in the word
vowel_indices = {0, 4, 8, 14, 20}  # a=0, e=4, i=8, o=14, u=20
x1_vowel_count = np.array([
    sum(1 for letter in word if letter in vowel_indices)
    for word in words
])

# Random variable X2: alphabetic position of first letter (1-indexed)
x2_first_letter = words[:, 0] + 1  # 1 to 26

# Random variable X3: does the word start with a common CV-related letter? (fun proxy)
# Using 'c','v','i','p','f' as "CV letters" — a simple binary event
cv_letters = {2, 21, 8, 15, 5}  # c=2, v=21, i=8, p=15, f=5
x3_is_cv = np.array([1 if w[0] in cv_letters else 0 for w in words])

# Theoretical distribution for X1: Binomial(n=3, p=5/26)
p_vowel = 5 / 26
k_vals = np.array([0, 1, 2, 3])
theoretical_pmf = binom.pmf(k_vals, n=3, p=p_vowel)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Same Random Process (3-letter word source) — Three Different Random Variables',
    fontsize=13, fontweight='bold'
)

# X1: vowel count — Binomial
ax = axes[0]
unique, counts = np.unique(x1_vowel_count, return_counts=True)
ax.bar(unique, counts / N_TRIALS, color=COLORS['primary'], alpha=0.7,
       label='Observed (10,000 trials)')
ax.plot(k_vals, theoretical_pmf, 'o-', color=COLORS['highlight'],
        markersize=8, linewidth=2, label='Binomial(n=3, p=5/26)')
ax.set_title('X₁ = Number of Vowels\n→ Binomial distribution', fontsize=11)
ax.set_xlabel('Vowel count')
ax.set_ylabel('Probability')
ax.set_xticks([0, 1, 2, 3])
ax.legend(fontsize=8)

# X2: first letter position — approximately Uniform(1, 26)
ax = axes[1]
ax.hist(x2_first_letter, bins=26, range=(0.5, 26.5), density=True,
        color=COLORS['secondary'], alpha=0.7)
ax.axhline(1/26, color=COLORS['highlight'], linewidth=2,
           linestyle='--', label='Uniform(1/26)')
ax.set_title('X₂ = Position of First Letter\n→ Uniform distribution', fontsize=11)
ax.set_xlabel('Letter position (a=1, z=26)')
ax.set_ylabel('Probability')
ax.legend(fontsize=8)

# X3: starts with a CV letter — Bernoulli
ax = axes[2]
p_cv = len(cv_letters) / 26
observed_p = x3_is_cv.mean()
ax.bar([0, 1], [1 - observed_p, observed_p], color=COLORS['gradient'], alpha=0.7,
       label='Observed')
ax.plot([0, 1], [1 - p_cv, p_cv], 'o', color=COLORS['highlight'],
        markersize=12, label=f'Bernoulli(p={p_cv:.3f})')
ax.set_title('X₃ = Starts with a CV letter?\n→ Bernoulli distribution', fontsize=11)
ax.set_xlabel('Outcome (0=No, 1=Yes)')
ax.set_ylabel('Probability')
ax.set_xticks([0, 1])
ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Three questions about the same word — three different distributions.")
print(f"  X1 (vowels):       Binomial(n=3, p=5/26) — mean={3*p_vowel:.2f}, std={np.sqrt(3*p_vowel*(1-p_vowel)):.2f}")
print(f"  X2 (first letter): Uniform(1,26)          — mean={x2_first_letter.mean():.1f}")
print(f"  X3 (CV letter):    Bernoulli(p={p_cv:.3f})   — observed p={observed_p:.3f}")
print()
print("The random variable is the bridge: it asks a specific question about each outcome.")
print("The distribution is the answer: the complete shape of probabilities over all values.")
print()

# ── Algorithm ──────────────────────────────────────────────────────────────
# Step 2 — Parameters compress the distribution to a few numbers
# 1. Show that Binomial(n=3, p=5/26) is fully specified by just n and p
# 2. Vary n and p to show how the shape changes
# 3. Annotate with the parameter values on each subplot
# What to look for: two numbers determine the entire distribution shape.
#   This is the power of parametric models.
# ──────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Parameters Compress the Distribution — Two Numbers Determine Everything',
             fontsize=12, fontweight='bold')

scenarios = [
    (3,  5/26,  'Word source\n(n=3, p=5/26)'),
    (6,  5/26,  '6-letter word\n(n=6, p=5/26)'),
    (3,  10/26, 'More vowel-like alphabet\n(n=3, p=10/26)'),
]

for ax, (n, p, title) in zip(axes, scenarios):
    k = np.arange(0, n + 1)
    pmf = binom.pmf(k, n, p)
    ax.bar(k, pmf, color=COLORS['primary'], alpha=0.7)
    ax.axvline(n * p, color=COLORS['highlight'], linewidth=2,
               linestyle='--', label=f'mean = np = {n*p:.2f}')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Vowel count')
    ax.set_ylabel('P(X = k)')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

print("Changing n (word length) widens and shifts the distribution.")
print("Changing p (vowel probability) skews it left or right.")
print("Two numbers. Complete control over the entire shape.")
print()


# ── Algorithm ──────────────────────────────────────────────────────────────
# The Data Manifold — why images live on a thin surface in high-dimensional space
# 1. Generate a Swiss roll: a 2D manifold curved and embedded in 3D space
#    This is the analogy for natural images in pixel space —
#    data lives on a low-dimensional curved surface inside a vast ambient space
# 2. Show the full 3D ambient space (mostly empty)
# 3. Colour points by their position on the manifold to show its 2D structure
# 4. Second plot: "unroll" the manifold — show it is actually 2D (like an encoder does)
# What to look for:
#   - The Swiss roll fills almost none of the 3D box — most of the space is empty
#   - Every point on it can be described by just 2 numbers (t, height) not 3
#   - Unrolling = what an encoder does: map the curved surface to a flat 2D space
#   - In CV: replace 3D → 150,528-d pixel space, 2D → d-d feature/latent space
# ──────────────────────────────────────────────────────────────────────────

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(42)
n_points = 2000

# Generate Swiss roll: a 2D manifold embedded in 3D
# Parameter t controls position along the roll (the "intrinsic" coordinate)
t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_points))
height = np.random.rand(n_points) * 10  # second intrinsic coordinate

# Embed in 3D (the "pixel space" analogy)
x_3d = t * np.cos(t)
y_3d = height
z_3d = t * np.sin(t)

# Normalise for display
x_3d = (x_3d - x_3d.mean()) / x_3d.std()
z_3d = (z_3d - z_3d.mean()) / z_3d.std()
y_3d = (y_3d - y_3d.mean()) / y_3d.std()

# Colour by t — reveals the 2D structure of the manifold
colours = plt.cm.viridis((t - t.min()) / (t.max() - t.min()))

fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    'The Data Manifold — Natural Images Occupy a Thin Curved Surface in Pixel Space',
    fontsize=13, fontweight='bold'
)

# Left: 3D ambient space with the manifold inside it
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.scatter(x_3d, y_3d, z_3d, c=colours, s=8, alpha=0.6)
ax1.set_title('3D ambient space\n(analogy: pixel space)', fontsize=10)
ax1.set_xlabel('dim 1')
ax1.set_ylabel('dim 2')
ax1.set_zlabel('dim 3')
ax1.text2D(0.05, 0.92,
    'Data lives on\nthis thin surface.\nMost of the box\nis empty.',
    transform=ax1.transAxes, fontsize=8, color='#F44336',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Middle: same manifold from a different angle — shows it is a surface not a volume
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(x_3d, y_3d, z_3d, c=colours, s=8, alpha=0.6)
ax2.view_init(elev=15, azim=45)
ax2.set_title('Same manifold, different angle\n(2D structure visible in colours)', fontsize=10)
ax2.set_xlabel('dim 1')
ax2.set_ylabel('dim 2')
ax2.set_zlabel('dim 3')

# Right: unrolled — the encoder maps the curved surface to flat 2D (latent space)
ax3 = fig.add_subplot(1, 3, 3)
scatter = ax3.scatter(t, height, c=colours, s=8, alpha=0.6)
ax3.set_title('Unrolled manifold — latent space\n(what an encoder produces)', fontsize=10)
ax3.set_xlabel('Intrinsic coordinate 1  (learned by encoder)')
ax3.set_ylabel('Intrinsic coordinate 2  (learned by encoder)')
ax3.text(0.05, 0.92,
    'Same data, now flat.\nTwo numbers describe\nevery point.',
    transform=ax3.transAxes, fontsize=8, color='#2196F3',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("The Swiss roll is a 2D manifold embedded in 3D space.")
print("Natural images are a ~low-d manifold embedded in 150,528-d pixel space.")
print()
print("Key insight:")
print("  Ambient space dimension  = pixel space dimension  (e.g. 150,528)")
print("  Manifold dimension       = intrinsic structure     (e.g. ~100s)")
print("  Encoder output dimension = latent space            (e.g. 128–2048)")
print()
print("The encoder's job: find the two (or d) numbers that describe each point on the manifold.")
print("The decoder's job: map those numbers back to a valid image (point on the manifold).")

# ── Algorithm ──────────────────────────────────────────────────────────────
# Option 2 — Interactive 3D manifold with Plotly
# 1. Generate Swiss roll data (same as above)
# 2. Plot as rotatable 3D scatter in plotly — student spins it in browser
# 3. Side-by-side with unrolled 2D view
# 4. Save as manifold_interactive.html next to this script
# What to look for: rotate the 3D view to feel the curvature;
#   compare with the flat 2D unrolled version — same colours, now flat
# ──────────────────────────────────────────────────────────────────────────

import plotly.graph_objects as go
from plotly.subplots import make_subplots

np.random.seed(42)
n_roll = 1500
t_roll = 1.5 * np.pi * (1 + 2 * np.random.rand(n_roll))
h_roll = np.random.rand(n_roll) * 10
x_roll = (t_roll * np.cos(t_roll)); x_roll = (x_roll - x_roll.mean()) / x_roll.std()
y_roll = (h_roll - h_roll.mean()) / h_roll.std()
z_roll = (t_roll * np.sin(t_roll)); z_roll = (z_roll - z_roll.mean()) / z_roll.std()
color_vals = (t_roll - t_roll.min()) / (t_roll.max() - t_roll.min())
t_norm = color_vals
h_norm = (h_roll - h_roll.min()) / (h_roll.max() - h_roll.min())

fig_plotly = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
    subplot_titles=(
        "Pixel space — curved manifold in 3D (rotate me!)",
        "Latent space — encoder unrolls to 2D"
    )
)

fig_plotly.add_trace(go.Scatter3d(
    x=x_roll, y=y_roll, z=z_roll, mode="markers",
    marker=dict(size=3, color=color_vals, colorscale="Viridis", opacity=0.7),
    name="Pixel space",
    hovertemplate="Intrinsic coord: %{marker.color:.2f}<extra></extra>"
), row=1, col=1)

fig_plotly.add_trace(go.Scatter(
    x=t_norm, y=h_norm, mode="markers",
    marker=dict(size=4, color=color_vals, colorscale="Viridis", opacity=0.7),
    name="Latent space",
    hovertemplate="z1: %{x:.2f}  z2: %{y:.2f}<extra></extra>"
), row=1, col=2)

fig_plotly.update_layout(
    title=dict(
        text="The Data Manifold: Pixel Space vs Latent Space<br>"
             "<sup>Left: 3D curved surface — rotate to explore. "
             "Right: encoder unrolls it flat. Same colours = same points.</sup>",
        x=0.5
    ),
    height=550, showlegend=False
)
fig_plotly.update_xaxes(title_text="Latent coordinate 1", row=1, col=2)
fig_plotly.update_yaxes(title_text="Latent coordinate 2", row=1, col=2)

_html_path = Path(__file__).parent / "manifold_interactive.html"
fig_plotly.write_html(str(_html_path))
print(f"\nInteractive manifold saved → {_html_path}")
print("Open manifold_interactive.html in your browser and rotate the 3D view.")



# ── Algorithm ──────────────────────────────────────────────
# Load the cameraman image and extract two patches
# 1. Load cameraman.tif as a grayscale array (0–255)
# 2. Define two 48×48 regions:
#    - Patch A: flat sky (top-right) — uniform brightness
#    - Patch B: textured foreground (camera body) — high contrast
# 3. Print mean and std of each patch
# 4. Display full image with patch locations + histograms side by side
# What to look for: both patches have ~identical means (~161) but wildly
#   different standard deviations. The histogram shape distinguishes them.
# ───────────────────────────────────────────────────────────

# Load as grayscale — every pixel is a single brightness value (0–255)
cameraman_image = np.array(
    Image.open(Path(__file__).parent / '../../DIP3E_Original_Images_CH02/Fig0222(b)(cameraman).tif').convert('L')
)

# Two 48×48 patches with nearly identical means but very different structure
PATCH_SIZE = 48

# Patch A: flat sky region (top-right) — uniform, low noise
flat_row, flat_col = 36, 188
patch_flat = cameraman_image[flat_row : flat_row + PATCH_SIZE,
                             flat_col : flat_col + PATCH_SIZE]

# Patch B: textured foreground (camera body area) — lots of variation
textured_row, textured_col = 26, 48
patch_textured = cameraman_image[textured_row : textured_row + PATCH_SIZE,
                                 textured_col : textured_col + PATCH_SIZE]

print(f'Patch A (flat sky):   mean = {patch_flat.mean():.1f},  std = {patch_flat.std():.1f}')
print(f'Patch B (textured):   mean = {patch_textured.mean():.1f},  std = {patch_textured.std():.1f}')
print(f'Means differ by only: {abs(patch_flat.mean() - patch_textured.mean()):.1f} gray levels')

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Same Mean Brightness — Completely Different Story', fontsize=14, fontweight='bold')

ax_full = axes[0, 0]
ax_full.imshow(cameraman_image, cmap='gray', vmin=0, vmax=255)
rect_flat = mpatches.Rectangle(
    (flat_col, flat_row), PATCH_SIZE, PATCH_SIZE,
    linewidth=2, edgecolor='#2196F3', facecolor='none', label='Patch A (flat)'
)
rect_textured = mpatches.Rectangle(
    (textured_col, textured_row), PATCH_SIZE, PATCH_SIZE,
    linewidth=2, edgecolor='#FF5722', facecolor='none', label='Patch B (textured)'
)
ax_full.add_patch(rect_flat)
ax_full.add_patch(rect_textured)
ax_full.legend(loc='lower right', fontsize=9)
ax_full.set_title('Full Image — Two Patches Selected')
ax_full.axis('off')

axes[0, 1].imshow(patch_flat, cmap='gray', vmin=0, vmax=255)
axes[0, 1].set_title(f'Patch A — Flat Sky\nmean = {patch_flat.mean():.1f},  std = {patch_flat.std():.1f}', color='#2196F3')
axes[0, 1].axis('off')

axes[0, 2].imshow(patch_textured, cmap='gray', vmin=0, vmax=255)
axes[0, 2].set_title(f'Patch B — Textured Foreground\nmean = {patch_textured.mean():.1f},  std = {patch_textured.std():.1f}', color='#FF5722')
axes[0, 2].axis('off')

axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5,
    'Same mean.\nDifferent\ndistributions.',
    ha='center', va='center', fontsize=16, fontweight='bold',
    transform=axes[1, 0].transAxes
)

axes[1, 1].hist(patch_flat.ravel(), bins=30, range=(0, 255),
                color='#2196F3', alpha=0.8, density=True)
axes[1, 1].axvline(patch_flat.mean(), color='black', linestyle='--', linewidth=1.5, label=f'mean={patch_flat.mean():.1f}')
axes[1, 1].set_title('Patch A Pixel Distribution')
axes[1, 1].set_xlabel('Pixel intensity (0–255)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].set_xlim(0, 255)

axes[1, 2].hist(patch_textured.ravel(), bins=30, range=(0, 255),
                color='#FF5722', alpha=0.8, density=True)
axes[1, 2].axvline(patch_textured.mean(), color='black', linestyle='--', linewidth=1.5, label=f'mean={patch_textured.mean():.1f}')
axes[1, 2].set_title('Patch B Pixel Distribution')
axes[1, 2].set_xlabel('Pixel intensity (0–255)')
axes[1, 2].set_ylabel('Density')
axes[1, 2].legend()
axes[1, 2].set_xlim(0, 255)

plt.tight_layout()
plt.show()

print('\nKey takeaway:')
print('  The mean alone cannot distinguish a flat sensor region from a textured one.')
print('  You need the full distribution — the shape of the histogram — to understand the data.')

# ── Algorithm ──────────────────────────────────────────────
# Patch histograms as probability distributions → generation
# Part 1 — approximation:
# 1. Compute the full-image pixel histogram (ground truth distribution)
# 2. Sample 4 random 48×48 patches
# 3. For each patch: plot its histogram against the full-image histogram
# Part 2 — generation:
# 1. Normalise one patch histogram into a PMF
# 2. Sample new pixel values from that PMF with np.random.choice
# 3. Reshape into a PATCH_SIZE × PATCH_SIZE image
# What to look for: generated patch has right statistics but looks like noise —
#   the histogram captures WHAT values appear but not WHERE.
# ───────────────────────────────────────────────────────────

np.random.seed(0)

full_image_hist, bin_edges = np.histogram(
    cameraman_image.ravel(), bins=64, range=(0, 255), density=True
)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle(
    'A Patch Histogram ≈ A Probability Distribution\n'
    'Sample from it → generate new image patches',
    fontsize=13, fontweight='bold'
)

for col in range(4):
    rand_row = np.random.randint(0, cameraman_image.shape[0] - PATCH_SIZE)
    rand_col = np.random.randint(0, cameraman_image.shape[1] - PATCH_SIZE)
    patch = cameraman_image[rand_row : rand_row + PATCH_SIZE, rand_col : rand_col + PATCH_SIZE]

    patch_hist, _ = np.histogram(patch.ravel(), bins=64, range=(0, 255), density=True)

    axes[0, col].imshow(patch, cmap='gray', vmin=0, vmax=255)
    axes[0, col].set_title(f'Patch at ({rand_row}, {rand_col})', fontsize=9)
    axes[0, col].axis('off')

    axes[1, col].bar(
        bin_centers, patch_hist, width=bin_width,
        color=COLORS['secondary'], alpha=0.6, label='Patch histogram'
    )
    axes[1, col].plot(
        bin_centers, full_image_hist,
        color='black', linewidth=1.8, alpha=0.7, label='Full image (ground truth)'
    )
    axes[1, col].set_xlabel('Pixel intensity')
    axes[1, col].set_ylabel('Density')
    axes[1, col].set_xlim(0, 255)
    axes[1, col].legend(fontsize=7)

plt.tight_layout()
plt.show()

np.random.seed(1)
example_row = np.random.randint(0, cameraman_image.shape[0] - PATCH_SIZE)
example_col = np.random.randint(0, cameraman_image.shape[1] - PATCH_SIZE)
example_patch = cameraman_image[example_row : example_row + PATCH_SIZE, example_col : example_col + PATCH_SIZE]

patch_hist_norm, bin_edges_norm = np.histogram(example_patch.ravel(), bins=64, range=(0, 255), density=False)
patch_probs = patch_hist_norm / patch_hist_norm.sum()
bin_centers_norm = (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2

sampled_pixel_values = np.random.choice(
    bin_centers_norm.astype(int), size=PATCH_SIZE * PATCH_SIZE, p=patch_probs
)
generated_patch = sampled_pixel_values.reshape(PATCH_SIZE, PATCH_SIZE).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Generation: Sample from the Patch Distribution → New Patch', fontsize=12, fontweight='bold')

axes[0].imshow(example_patch, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Original patch')
axes[0].axis('off')

axes[1].bar(bin_centers_norm, patch_probs, width=bin_width,
            color=COLORS['primary'], alpha=0.8)
axes[1].set_title('Learned distribution\n(patch histogram)')
axes[1].set_xlabel('Pixel intensity')
axes[1].set_ylabel('Probability')
axes[1].set_xlim(0, 255)

axes[2].imshow(generated_patch, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Generated patch\n(sampled from distribution)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print('The generated patch has the same pixel statistics as the original...')
print(f'  Original  — mean: {example_patch.mean():.1f},  std: {example_patch.std():.1f}')
print(f'  Generated — mean: {generated_patch.mean():.1f},  std: {generated_patch.std():.1f}')
print()
print('But it looks like noise — because the histogram discards ALL spatial structure.')
print('Generative models (VAEs, GANs, diffusion) learn distributions that PRESERVE spatial structure.')
print('That is the journey from this histogram to a neural generative model.')

# ── Algorithm ──────────────────────────────────────────────
# Simulate two patches with identical means but different spreads
# 1. Draw 10,000 pixels for Patch A from Normal(mean=128, std=3) — tight texture
# 2. Draw 10,000 pixels for Patch B from Normal(mean=128, std=25) — wide texture
# 3. Clip both to [0, 255]
# 4. Confirm means are identical
# What to look for: same mean, wildly different std
# ───────────────────────────────────────────────────────────

np.random.seed(42)
number_of_pixels = 10000

patch_a_values = np.random.normal(loc=128, scale=3, size=number_of_pixels).clip(0, 255)
patch_b_values = np.random.normal(loc=128, scale=25, size=number_of_pixels).clip(0, 255)

print(f"Sensor A: mean = {patch_a_values.mean():.1f}, std = {patch_a_values.std():.1f}")
print(f"Sensor B: mean = {patch_b_values.mean():.1f}, std = {patch_b_values.std():.1f}")
print()
print("Same mean. Completely different texture / content.")
print("The MEAN tells you nothing about the noise. The DISTRIBUTION tells you everything.")

# ── Algorithm ──────────────────────────────────────────────
# Visualize why the mean alone is not enough
# 1. Plot histogram of Patch A — narrow spike around 128
# 2. Plot histogram of Patch B — wide spread across [0, 255]
# 3. Overlay the mean line on both — same position, different shapes
# 4. Third panel: 200-pixel row of each patch overlaid
# What to look for: the dashed mean line sits at the same x-position,
#   but the distribution shapes are completely different.
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(patch_a_values, bins=80, density=True, color=COLORS['primary'], alpha=0.7)
axes[0].axvline(patch_a_values.mean(), color='k', linestyle='--', linewidth=2, label=f'mean = {patch_a_values.mean():.0f}')
axes[0].set_title(f'Patch A (uniform texture)\nσ = {patch_a_values.std():.1f}', fontsize=12)
axes[0].set_xlabel('Pixel value')
axes[0].set_ylabel('Density')
axes[0].set_xlim(0, 255)
axes[0].legend()

axes[1].hist(patch_b_values, bins=80, density=True, color=COLORS['highlight'], alpha=0.7)
axes[1].axvline(patch_b_values.mean(), color='k', linestyle='--', linewidth=2, label=f'mean = {patch_b_values.mean():.0f}')
axes[1].set_title(f'Patch B (varied texture)\nσ = {patch_b_values.std():.1f}', fontsize=12)
axes[1].set_xlabel('Pixel value')
axes[1].set_xlim(0, 255)
axes[1].legend()

row_length = 200
axes[2].plot(patch_a_values[:row_length], color=COLORS['primary'], linewidth=0.8, label='Patch A', alpha=0.8)
axes[2].plot(patch_b_values[:row_length], color=COLORS['highlight'], linewidth=0.8, label='Patch B', alpha=0.8)
axes[2].axhline(128, color='k', linestyle='--', linewidth=1, label='True value (128)')
axes[2].set_xlabel('Pixel position')
axes[2].set_ylabel('Pixel value')
axes[2].set_title('One Row of Pixels: Same Mean, Different Noise', fontsize=12)
axes[2].legend(fontsize=9)
axes[2].set_ylim(0, 255)

plt.tight_layout()
plt.show()

print("The mean (128) is identical. But Patch B is useless for inspection —")
print("the noise is so large that real signal differences would be buried in randomness.")
print()
print("The DISTRIBUTION captures this difference. The mean alone cannot.")

# ── Algorithm ──────────────────────────────────────────────
# How distribution shape encodes image content
# 1. Simulate four region types:
#    - Uniform texture: Normal(150, 8)
#    - Dark region: Normal(30, 5) clipped at 0 — skewed right
#    - Bright highlight with hot pixels: Normal(128, 5) + 1% spikes
#    - Edge region: mixture of Normal(80, 5) and Normal(180, 5) — bimodal
# 2. Plot each histogram and annotate with shape features
# What to look for: each shape fingerprints a different physical process
# ───────────────────────────────────────────────────────────

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

signal_1 = np.random.normal(150, 8, size=10000)
axes[0, 0].hist(signal_1, bins=60, density=True, color=COLORS['primary'], alpha=0.7)
axes[0, 0].set_title('Uniform texture patch (mid-tone)\nSymmetric, narrow → low variation', fontsize=11)
axes[0, 0].set_xlabel('Pixel value')

dark_signal = np.random.normal(8, 6, size=10000)
dark_signal_clipped = np.clip(dark_signal, 0, 255)
axes[0, 1].hist(dark_signal_clipped, bins=60, density=True, color=COLORS['highlight'], alpha=0.7)
axes[0, 1].set_title('Dark region patch (near noise floor)\nSkewed right — clipped at 0', fontsize=11)
axes[0, 1].set_xlabel('Pixel value')

normal_signal = np.random.normal(128, 5, size=10000)
hot_pixel_mask = np.random.random(10000) < 0.01
normal_signal[hot_pixel_mask] = np.random.uniform(200, 255, size=hot_pixel_mask.sum())
axes[1, 0].hist(normal_signal, bins=80, density=True, color=COLORS['gradient'], alpha=0.7)
axes[1, 0].set_title('Bright highlight patch with sensor defect\nHeavy tail on the right → rare spikes', fontsize=11)
axes[1, 0].set_xlabel('Pixel value')

edge_signal = np.concatenate([
    np.random.normal(80, 5, size=6000),
    np.random.normal(180, 5, size=4000),
])
np.random.shuffle(edge_signal)
axes[1, 1].hist(edge_signal, bins=60, density=True, color=COLORS['secondary'], alpha=0.7)
axes[1, 1].set_title('Edge region patch (dark background + bright object)\nBimodal — two distinct intensity states', fontsize=11)
axes[1, 1].set_xlabel('Pixel value')

for ax in axes.flat:
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.suptitle('Distribution Shape = Image Content / Physical Process', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("Each distribution shape tells a DIFFERENT story about the image content or noise process.")
print("You cannot distinguish these four situations from their means alone.")

# ── Algorithm ──────────────────────────────────────────────
# Watch a histogram converge to the true distribution
# 1. Fix true distribution: Poisson(λ=30)
# 2. Draw samples of increasing size: n = 10, 50, 200, 10,000
# 3. For each: plot empirical histogram and theoretical PMF
# 4. Show CDF convergence in bottom row
# What to look for: at n=10 the histogram is noisy; at n=10,000 it matches
#   the PMF almost exactly.
# ───────────────────────────────────────────────────────────

expected_intensity = 30

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
sample_sizes = [10, 50, 200, 10000]

for col, n_samples in enumerate(sample_sizes):
    samples = np.random.poisson(expected_intensity, size=n_samples)

    ax = axes[0, col]
    k_range = np.arange(10, 55)
    ax.hist(samples, bins=k_range - 0.5, density=True, color=COLORS['primary'],
            alpha=0.6, label='Measured')

    theoretical_pmf = poisson.pmf(k_range, expected_intensity)
    ax.plot(k_range, theoretical_pmf, 'o-', color=COLORS['highlight'], markersize=3,
            linewidth=2, label='Theoretical\nPoisson(λ=30)')

    ax.set_title(f'n = {n_samples} samples', fontsize=12, fontweight='bold')
    ax.set_xlabel('Pixel intensity')
    if col == 0:
        ax.set_ylabel('Probability')
    ax.legend(fontsize=7)
    ax.set_ylim(0, 0.15)

    ax = axes[1, col]
    ax.hist(samples, bins=k_range - 0.5, density=True, cumulative=True,
            color=COLORS['primary'], alpha=0.6, label='Measured CDF')
    ax.plot(k_range, poisson.cdf(k_range, expected_intensity), '-',
            color=COLORS['highlight'], linewidth=2, label='Theoretical CDF')
    ax.set_xlabel('Pixel intensity')
    if col == 0:
        ax.set_ylabel('Cumulative probability')
    ax.legend(fontsize=7)

plt.suptitle('Histogram → Distribution: More Samples = Better Estimate of the True Pattern',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("With 10 samples: the histogram is noisy, barely resembles the theoretical shape")
print("With 10,000 samples: the histogram IS the distribution (to within statistical noise)")
print()
print("The theoretical distribution is the 'infinite data' limit of the histogram.")
print("It's the model — the idealized pattern that your measurements approximate.")

# ── Algorithm ──────────────────────────────────────────────
# Fit a distribution to data and recover it from one number
# 1. Set true λ = 75 (unknown)
# 2. Generate 5,000 measurements from Poisson(75)
# 3. Estimate λ = sample mean (MLE for Poisson)
# 4. Reconstruct the full PMF from that one number
# 5. Overlay reconstructed PMF on the data histogram
# What to look for: one number (λ) reconstructs the entire distribution
# ───────────────────────────────────────────────────────────

np.random.seed(42)

true_lambda = 75
measurements = np.random.poisson(true_lambda, size=5000)
estimated_lambda = measurements.mean()

print(f"True λ (unknown): {true_lambda}")
print(f"Estimated λ (from {len(measurements)} measurements): {estimated_lambda:.2f}")
print(f"That ONE number reconstructs the entire distribution:")
print()

fig, ax = plt.subplots(figsize=(10, 5))

k_range = np.arange(measurements.min(), measurements.max() + 1)
ax.hist(measurements, bins=k_range - 0.5, density=True, color=COLORS['primary'],
        alpha=0.5, label=f'Measured data (5000 samples)')
ax.plot(k_range, poisson.pmf(k_range, estimated_lambda), 'o-',
        color=COLORS['highlight'], markersize=4, linewidth=2,
        label=f'Poisson(λ={estimated_lambda:.1f}) — reconstructed from 1 number')

ax.set_xlabel('Pixel intensity', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
ax.set_title(f'One Parameter (λ = {estimated_lambda:.1f}) Reconstructs the Entire Distribution', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"From λ = {estimated_lambda:.1f} alone, we know:")
print(f"  Mean photon count: {estimated_lambda:.1f}")
print(f"  Noise (std dev): √{estimated_lambda:.1f} = {np.sqrt(estimated_lambda):.1f} photons")
print(f"  SNR: {estimated_lambda / np.sqrt(estimated_lambda):.1f}")
print(f"  P(count < 60): {poisson.cdf(60, estimated_lambda):.3f} ({poisson.cdf(60, estimated_lambda)*100:.1f}%)")
print(f"  P(count > 90): {1 - poisson.cdf(90, estimated_lambda):.3f} ({(1-poisson.cdf(90, estimated_lambda))*100:.1f}%)")
print()
print("One number → full noise model → all predictions. That's the power of distributions.")

# ── Algorithm ──────────────────────────────────────────────────────────────
# Option 3 — t-SNE on real cameraman patches
# 1. Extract 600 random 16x16 patches from the cameraman image
# 2. Flatten each: 256-d pixel vectors
# 3. Run t-SNE: project from 256-d down to 2-d
# 4. Plot coloured by mean brightness and by std (texture)
# 5. Overlay actual patch thumbnails at 50 positions
# What to look for:
#   - Dark patches cluster together, bright patches cluster together
#   - Uniform patches (low std) separate from textured/edge patches (high std)
#   - This is the manifold hypothesis on real image data
# ──────────────────────────────────────────────────────────────────────────

from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

np.random.seed(42)
TSNE_PATCH = 16
N_PATCHES  = 600
img_h, img_w = cameraman_image.shape

patches_flat, patch_imgs = [], []
for _ in range(N_PATCHES):
    r = np.random.randint(0, img_h - TSNE_PATCH)
    c = np.random.randint(0, img_w - TSNE_PATCH)
    p = cameraman_image[r:r+TSNE_PATCH, c:c+TSNE_PATCH]
    patches_flat.append(p.ravel().astype(np.float32) / 255.0)
    patch_imgs.append(p)

patches_array = np.array(patches_flat)

print(f"\nRunning t-SNE on {N_PATCHES} patches ({TSNE_PATCH}x{TSNE_PATCH}={TSNE_PATCH**2}-d) ...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
embedding = tsne.fit_transform(patches_array)
print("t-SNE done.")

patch_means = patches_array.mean(axis=1)
patch_stds  = patches_array.std(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(
    "t-SNE: Real Cameraman Patches — 256-d Pixel Space → 2-d Embedding\n"
    "Similar patches cluster together: the manifold hypothesis on real image data",
    fontsize=12, fontweight="bold"
)
sc1 = axes[0].scatter(embedding[:,0], embedding[:,1],
                      c=patch_means, cmap="gray", s=15, alpha=0.8)
plt.colorbar(sc1, ax=axes[0], label="Patch mean brightness")
axes[0].set_title("Coloured by mean brightness\nDark / bright regions form separate clusters")
axes[0].set_xlabel("t-SNE dim 1"); axes[0].set_ylabel("t-SNE dim 2")

sc2 = axes[1].scatter(embedding[:,0], embedding[:,1],
                      c=patch_stds, cmap="hot", s=15, alpha=0.8)
plt.colorbar(sc2, ax=axes[1], label="Patch std (texture)")
axes[1].set_title("Coloured by std (texture amount)\nUniform patches separate from edge/textured patches")
axes[1].set_xlabel("t-SNE dim 1")

plt.tight_layout()
plt.show()

# Thumbnail overlay
fig2, ax2 = plt.subplots(figsize=(14, 10))
ax2.scatter(embedding[:,0], embedding[:,1], c=patch_means, cmap="gray", s=8, alpha=0.25)
ax2.set_title(
    "Patch thumbnails in t-SNE space\n"
    "Each image = one 16x16 patch positioned by its 2-d embedding",
    fontsize=11, fontweight="bold"
)
ax2.axis("off")
for idx in np.linspace(0, N_PATCHES-1, 50).astype(int):
    img_box = OffsetImage(patch_imgs[idx], zoom=1.8, cmap="gray")
    ab = AnnotationBbox(img_box, (embedding[idx,0], embedding[idx,1]),
                        frameon=True, pad=0.1,
                        bboxprops=dict(edgecolor="steelblue", linewidth=0.5))
    ax2.add_artist(ab)
plt.tight_layout()
plt.show()

print("\nKey observation:")
print("  Dark uniform patches  → cluster together in the 2-d embedding")
print("  Bright uniform patches → separate cluster")
print("  Edge / textured patches → scattered (high variance, many modes)")
print("  This is the manifold hypothesis: 256-d pixel vectors projected to 2-d,")
print("  and semantic structure emerges automatically.")
