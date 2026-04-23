"""
Part 6: Putting It All Together — The Complete Image Noise Model

Assembles all distributions from Parts 1–5 into a full sensor simulation,
visualizes each stage of the signal chain, maps the three noise regimes,
and simulates a complete gradient image capture.

What this script demonstrates:
  - Full sensor signal chain: photons → electrons → ADC → pixel values
  - Each noise source (shot, dark current, read, quantization) as a distribution
  - The CLT at work: Poisson + Gaussian sources converge to Gaussian total
  - Three noise regimes on a log-log noise budget plot
  - SNR as a function of signal level across the full dynamic range
  - A synthetic gradient image capturing all three regimes in one frame

Run: python part6_putting_it_together.py
"""

# --- Setup ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

COLORS = {
    'primary': '#2196F3',
    'secondary': '#4CAF50',
    'result': '#FFC107',
    'highlight': '#F44336',
    'transform': '#9C27B0',
    'gradient': '#FF9800',
}

np.random.seed(42)
print("Setup complete.")


# ── Algorithm ──────────────────────────────────────────────
# Define the complete image noise simulation function
# Full signal chain from photons to pixel values:
# 1. Photon arrival: Poisson(λ) — quantum noise floor
# 2. Quantum efficiency: Binomial(n_photons, QE) — not every photon is detected
# 3. Dark current: Poisson(dark_current) — thermal electrons even in the dark
# 4. Total electrons clipped at full_well_capacity
# 5. Read noise: Normal(0, read_noise_sigma) — readout circuit electronics
# 6. ADC quantization: scale to bit_depth and round
# Returns dict of intermediate signals so every stage can be inspected.
# What to look for: shape evolves from discrete Poisson to nearly Gaussian —
#   the CLT at work inside a single pixel.
# ───────────────────────────────────────────────────────────

def simulate_sensor(expected_photons, dark_current_electrons, read_noise_sigma,
                    full_well_capacity, bit_depth, quantum_efficiency=0.7,
                    num_exposures=10000):
    """Simulate the complete signal chain from photons to pixel values.

    This model underlies: image denoising (NLM, BM3D), HDR fusion, image restoration.
    Understanding this chain is what makes those algorithms work — they exploit the
    statistical structure of the noise at each stage.

    Returns dict with intermediate signals at every stage for inspection.
    """
    results = {}

    # Stage 1: Photon arrival (Poisson) — how many photons reach the sensor
    incident_photons = np.random.poisson(expected_photons, size=num_exposures)
    results['incident_photons'] = incident_photons

    # Stage 2: Quantum efficiency (Binomial) — how many get detected
    detected_electrons = np.random.binomial(incident_photons, quantum_efficiency)
    results['detected_electrons'] = detected_electrons

    # Stage 3: Dark current (Poisson) — thermal electrons (even in the dark)
    dark_electrons = np.random.poisson(dark_current_electrons, size=num_exposures)
    results['dark_electrons'] = dark_electrons

    # Stage 4: Total electrons in the well
    total_electrons = detected_electrons + dark_electrons

    # Stage 5: Saturation — clip at full well capacity
    total_electrons_clipped = np.clip(total_electrons, 0, full_well_capacity)
    results['total_electrons'] = total_electrons_clipped

    # Stage 6: Read noise (Gaussian) — added during readout
    read_noise = np.random.normal(0, read_noise_sigma, size=num_exposures)
    analog_signal = total_electrons_clipped + read_noise
    results['analog_signal'] = analog_signal

    # Stage 7: ADC — convert to digital values
    adc_levels = 2 ** bit_depth
    adc_step = full_well_capacity / adc_levels
    digital_values = np.clip(np.round(analog_signal / adc_step), 0, adc_levels - 1)
    results['digital_values'] = digital_values

    # Stage 8: Scale to 8-bit for display (if bit_depth > 8)
    pixel_values = digital_values / (adc_levels - 1) * 255
    results['pixel_values'] = pixel_values

    return results


# Simulate a "mid-tone" pixel: 500 expected photons, typical industrial camera
results = simulate_sensor(
    expected_photons=500,
    dark_current_electrons=10,
    read_noise_sigma=3.0,
    full_well_capacity=10000,
    bit_depth=12,
    quantum_efficiency=0.7,
    num_exposures=50000,
)

print("Signal chain for one pixel (50,000 exposures simulated):")
print(f"  Incident photons:    mean = {results['incident_photons'].mean():.1f}, "
      f"std = {results['incident_photons'].std():.1f}")
print(f"  Detected electrons:  mean = {results['detected_electrons'].mean():.1f}, "
      f"std = {results['detected_electrons'].std():.1f}")
print(f"  Dark electrons:      mean = {results['dark_electrons'].mean():.1f}, "
      f"std = {results['dark_electrons'].std():.1f}")
print(f"  Total electrons:     mean = {results['total_electrons'].mean():.1f}, "
      f"std = {results['total_electrons'].std():.1f}")
print(f"  After read noise:    mean = {results['analog_signal'].mean():.1f}, "
      f"std = {results['analog_signal'].std():.1f}")
print(f"  Digital values:      mean = {results['digital_values'].mean():.1f}, "
      f"std = {results['digital_values'].std():.1f}")
print(f"  Pixel values (8-bit): mean = {results['pixel_values'].mean():.1f}, "
      f"std = {results['pixel_values'].std():.1f}")

# ── Algorithm ──────────────────────────────────────────────
# Visualize each stage of the signal chain
# Plot histogram at each of the 6 pipeline stages + Gaussian fit overlay
# Stages: incident photons, detected electrons, dark current,
#         total electrons, after read noise, final digital values
# What to look for: shape evolves from discrete Poisson to nearly Gaussian
#   as noise sources accumulate — the CLT at work inside a single pixel.
# ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

stages = [
    ('incident_photons', 'Stage 1: Photon Arrival\n(Poisson)', COLORS['primary']),
    ('detected_electrons', 'Stage 2: After QE\n(Binomial filter)', COLORS['secondary']),
    ('dark_electrons', 'Stage 3: Dark Current\n(Poisson, small)', COLORS['transform']),
    ('total_electrons', 'Stage 4: Total Electrons\n(sum of above)', COLORS['gradient']),
    ('analog_signal', 'Stage 5: After Read Noise\n(+ Gaussian)', COLORS['highlight']),
    ('digital_values', 'Stage 6: After ADC\n(quantised)', 'gray'),
]

for ax, (key, title, color) in zip(axes.flat, stages):
    data = results[key]

    ax.hist(data, bins=80, density=True, color=color, alpha=0.6, label='Simulated')

    # Overlay Gaussian fit — how well does a Normal describe each stage?
    mu, sigma = data.mean(), data.std()
    x_range = np.linspace(data.min(), data.max(), 300)
    ax.plot(x_range, norm.pdf(x_range, mu, sigma), 'k--', linewidth=2,
            label=f'N({mu:.0f}, {sigma:.1f}²)')

    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Value')
    ax.legend(fontsize=8)

plt.suptitle('Complete Sensor Signal Chain: Each Stage Adds Noise', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

print("Key observations:")
print("  1. Shot noise (Poisson) dominates at this photon count")
print("  2. QE reduces the signal but preserves the Poisson shape")
print("  3. Dark current adds a small amount of extra Poisson noise")
print("  4. Read noise barely changes the shape (small σ relative to shot noise)")
print("  5. The total is well-approximated by Gaussian (by the CLT)")
print("  6. ADC quantisation creates discrete steps but doesn't change the envelope")

# ── Algorithm ──────────────────────────────────────────────
# Map the three noise regimes across signal levels
# 1. Sweep expected photons from 1 to 10,000 on a log scale
# 2. Compute variance contribution from each noise source:
#    - Shot noise: σ² = λ × QE (scales with signal)
#    - Dark noise: σ² = dark_current (constant)
#    - Read noise: σ² = read_noise² (constant)
# 3. Plot noise sources and SNR on log-log scale
# 4. Mark regime crossover points
# What to look for: three regimes where different sources dominate.
#   This tells you which denoising strategy is appropriate.
# ───────────────────────────────────────────────────────────

expected_photon_range = np.logspace(0, 4, 200)  # 1 to 10,000 photons

read_noise_sigma = 3.0
dark_current = 5.0
full_well = 10000
quantum_efficiency = 0.7

signal = expected_photon_range * quantum_efficiency
shot_noise_variance = signal         # Poisson: variance = mean
dark_noise_variance = dark_current   # constant
read_noise_variance = read_noise_sigma ** 2  # constant

total_noise = np.sqrt(shot_noise_variance + dark_noise_variance + read_noise_variance)
snr = signal / total_noise

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: noise sources
ax = axes[0]
ax.loglog(expected_photon_range, np.sqrt(shot_noise_variance),
          color=COLORS['primary'], linewidth=2, label='Shot noise (√signal)')
ax.loglog(expected_photon_range, np.full_like(expected_photon_range, np.sqrt(read_noise_variance)),
          color=COLORS['highlight'], linewidth=2, label=f'Read noise (σ={read_noise_sigma}e⁻)')
ax.loglog(expected_photon_range, np.full_like(expected_photon_range, np.sqrt(dark_noise_variance)),
          color=COLORS['transform'], linewidth=2, label=f'Dark current (σ={np.sqrt(dark_current):.1f}e⁻)')
ax.loglog(expected_photon_range, total_noise, 'k-', linewidth=2.5, label='Total noise')

crossover_shot_read = read_noise_sigma ** 2
ax.axvline(crossover_shot_read / quantum_efficiency, color='gray', linestyle=':', alpha=0.5)
ax.axvline(full_well / quantum_efficiency, color='gray', linestyle=':', alpha=0.5)

ax.fill_betweenx([0.1, 1000], 1, crossover_shot_read / quantum_efficiency,
                  alpha=0.05, color=COLORS['highlight'])
ax.fill_betweenx([0.1, 1000], crossover_shot_read / quantum_efficiency,
                  full_well / quantum_efficiency, alpha=0.05, color=COLORS['primary'])

ax.set_xlabel('Expected photons', fontsize=12)
ax.set_ylabel('Noise (electrons, std dev)', fontsize=12)
ax.set_title('Noise Sources vs Signal Level', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 500)

# Right: SNR
ax = axes[1]
ax.loglog(expected_photon_range, snr, 'k-', linewidth=2.5, label='Actual SNR')
ax.loglog(expected_photon_range, np.sqrt(signal), '--', color=COLORS['primary'],
          linewidth=1.5, label='Shot-noise limit (√signal)')
ax.axhline(1, color=COLORS['highlight'], linestyle=':', linewidth=1.5, label='SNR = 1')

ax.fill_betweenx([0.01, 10000], 1, crossover_shot_read / quantum_efficiency,
                  alpha=0.05, color=COLORS['highlight'], label='Read-noise limited')
ax.fill_betweenx([0.01, 10000], crossover_shot_read / quantum_efficiency,
                  full_well / quantum_efficiency, alpha=0.05, color=COLORS['primary'],
                  label='Shot-noise limited')

ax.set_xlabel('Expected photons', fontsize=12)
ax.set_ylabel('SNR', fontsize=12)
ax.set_title('Signal-to-Noise Ratio vs Signal Level', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Three regimes:")
print(f"  1. READ-NOISE LIMITED (< ~{crossover_shot_read/quantum_efficiency:.0f} photons):")
print(f"     Shot noise < read noise. Electronics dominate. SNR ∝ signal (linear)")
print(f"  2. SHOT-NOISE LIMITED (~{crossover_shot_read/quantum_efficiency:.0f} to ~{full_well/quantum_efficiency:.0f} photons):")
print(f"     Shot noise > read noise. Physics dominates. SNR ∝ √signal (square root)")
print(f"  3. SATURATION (> ~{full_well/quantum_efficiency:.0f} photons):")
print(f"     Well is full. Information is lost. SNR collapses.")
print()
print("Industrial cameras operate in regime 2 (shot-noise limited) by design:")
print("  → Use enough light to get out of regime 1")
print("  → Control exposure to avoid regime 3")

# ── Algorithm ──────────────────────────────────────────────
# Simulate a full gradient image covering all noise regimes
# 1. Create 100×400 image with exponentially increasing brightness (1→15,000 photons)
# 2. Apply full noise model to every pixel (simulate_sensor per column)
# 3. Display noiseless ground truth and noisy captured image
# 4. Plot pixel-wise SNR across columns
# What to look for: left edge is grainy (read-noise limited, low SNR),
#   middle shows smooth Poisson shot noise, right edge saturates to white.
#   This is exactly the input to NLM/BM3D denoising algorithms.
# ───────────────────────────────────────────────────────────

np.random.seed(42)

image_height = 100
image_width = 400

# Exponentially increasing brightness: covers all three noise regimes
column_indices = np.arange(image_width)
expected_photons_per_pixel = 1 * np.exp(column_indices / image_width * np.log(15000))
expected_photons_2d = np.tile(expected_photons_per_pixel, (image_height, 1))

quantum_efficiency = 0.7
full_well_capacity = 10000
read_noise_sigma = 3.0
dark_current_electrons = 5
bit_depth = 12

# Step 1: Photon detection (Poisson + QE)
detected_electrons = np.random.poisson(
    np.clip(expected_photons_2d * quantum_efficiency, 0, None).astype(int)
)

# Step 2: Dark current
dark = np.random.poisson(dark_current_electrons, size=(image_height, image_width))

# Step 3: Total electrons (clip at full well)
total = np.clip(detected_electrons + dark, 0, full_well_capacity)

# Step 4: Read noise
read_noise = np.random.normal(0, read_noise_sigma, size=(image_height, image_width))
analog = total + read_noise

# Step 5: ADC
adc_levels = 2 ** bit_depth
pixel_values = np.clip(np.round(analog / full_well_capacity * (adc_levels - 1)),
                       0, adc_levels - 1)

# Scale to 8-bit for display
display_image = (pixel_values / (adc_levels - 1) * 255).astype(np.uint8)

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# Top: the captured image
axes[0].imshow(display_image, cmap='gray', aspect='auto')
axes[0].set_title('Simulated Camera Capture: Gradient Scene', fontsize=13)
axes[0].set_ylabel('Rows')

# Mark the three regimes
crossover = np.searchsorted(expected_photons_per_pixel, read_noise_sigma**2 / quantum_efficiency)
saturation = np.searchsorted(expected_photons_per_pixel, full_well_capacity / quantum_efficiency)
for boundary, label, ha in [(crossover, 'Read→Shot', 'left'),
                             (saturation, 'Shot→Saturated', 'right')]:
    if boundary < image_width:
        axes[0].axvline(boundary, color=COLORS['highlight'], linewidth=2, linestyle='--')
        axes[0].text(boundary + 3, 5, label, color=COLORS['highlight'], fontsize=10,
                     fontweight='bold', ha=ha)

# Middle: cross-section showing noise
middle_row = display_image[image_height // 2, :]
axes[1].plot(column_indices, middle_row, color=COLORS['primary'], linewidth=0.5, alpha=0.7)
true_signal = np.clip(expected_photons_per_pixel * quantum_efficiency, 0, full_well_capacity)
true_pixel = true_signal / full_well_capacity * 255
axes[1].plot(column_indices, true_pixel, 'k-', linewidth=2, label='True signal')
axes[1].set_ylabel('Pixel value')
axes[1].set_title('Cross-Section: One Row of Pixels', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Bottom: local SNR estimated from the image
window = 20
local_means = np.array([middle_row[max(0, i-window):i+window].mean()
                         for i in range(image_width)])
local_stds = np.array([middle_row[max(0, i-window):i+window].std()
                        for i in range(image_width)])
local_snr = np.where(local_stds > 0, local_means / local_stds, 0)

axes[2].semilogy(column_indices, local_snr, color=COLORS['secondary'], linewidth=1.5)
axes[2].axhline(1, color=COLORS['highlight'], linestyle=':', label='SNR = 1')
axes[2].set_xlabel('Pixel position (left = dark, right = bright)')
axes[2].set_ylabel('Local SNR')
axes[2].set_title('Signal-to-Noise Ratio Across the Image', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Left side (dark): noisy, SNR low — read-noise limited")
print("Middle (mid-tone): moderate noise, SNR improving — shot-noise limited")
print("Right side (bright): clean until saturation clips to white")
