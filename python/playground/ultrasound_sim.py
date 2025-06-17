import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Basic material description (assignment units)
# ------------------------------------------------------------------
class Material:
    def __init__(self, name, *, thickness, mu_a, density, c):
        self.name = name
        self.Thickness = thickness   # [cm]
        self.MuA = mu_a              # [dB/(MHz*cm)]
        self.density = density       # [g/cm³]
        self.c = c                   # [m/s]
        self.Z = density * 1000 * c  # [kg/(m^2 s)]

# Use values from your table (assignment)
Fat    = Material('Fat',    thickness=0.5, mu_a=0.9,  density=0.6,  c=1476)
Muscle = Material('Muscle', thickness=1,   mu_a=0.54, density=0.9,  c=1580) # Muscle thickness is 1cm if tumor sits in center 
Tumor  = Material('Tumor',  thickness=1,   mu_a=0.76, density=0.8,  c=1564)
Nerve  = Material('Nerve',  thickness=0.5, mu_a=0.9,  density=0.9,  c=1630)

material_list = [Fat, Muscle, Tumor, Nerve]

# ------------------------------------------------------------------
# Helper coefficients
# ------------------------------------------------------------------
def transmission(m1, m2):
    return 2 * m1.Z / (m1.Z + m2.Z)

def reflection(m1, m2):
    if m1 is m2:
        return 1.0
    return abs(m2.Z - m1.Z) / (m2.Z + m1.Z)

# ------------------------------------------------------------------
# Imaging system and protocol parameters
# ------------------------------------------------------------------
frequencies = np.array([3, 4, 6, 9, 12, 15])  # MHz
num_freqs   = frequencies.size
I0 = 0.1  # Input acoustic intensity [W/cm^2] (from assignment)
electronic_noise_std = 0.001  # W/cm^2 (from assignment)
MC_REPS = 1000                # Monte Carlo reps per freq/depth for stats



# --- Pick average speed for path ---
# (Or, if needed, use weighted average based on your actual geometry!)
c_mean = np.mean([mat.c for mat in material_list])  # average speed in m/s

# --- Calculate axial resolution for each frequency ---
axial_resolution_mm = c_mean / (2 * frequencies * 1e6) * 1000  # result in mm

print("Frequency (MHz)   Axial Resolution (mm)")
for f, res in zip(frequencies, axial_resolution_mm):
    print(f"{f:14.1f}   {res:20.3f}")

# --- Optional: Plot axial resolution vs frequency ---
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(frequencies, axial_resolution_mm, marker="o")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Axial Resolution (mm)")
plt.title("Axial Resolution vs Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Geometry: sum of thicknesses (all in cm)
# ------------------------------------------------------------------
depth_cm = Fat.Thickness + Muscle.Thickness + Tumor.Thickness + Nerve.Thickness
steps = 200
z_cm = np.linspace(0.0, depth_cm, steps)  # cm
z_mm = z_cm * 10

# Which tissue at each depth?
material_at_z = np.full_like(z_cm, None, dtype=object)
for i, d in enumerate(z_cm):
    if d < Fat.Thickness:
        material_at_z[i] = Fat
    elif d < Fat.Thickness + Muscle.Thickness:
        material_at_z[i] = Muscle
    elif d < Fat.Thickness + Muscle.Thickness + Tumor.Thickness:
        material_at_z[i] = Tumor
    else:
        material_at_z[i] = Nerve

# For attenuation, sum alpha*thickness for each layer up to current z
layer_boundaries = np.cumsum([0, Fat.Thickness, Muscle.Thickness, Tumor.Thickness, Nerve.Thickness])

# ------------------------------------------------------------------
# For each freq, for each depth, simulate intensity including attenuation
# ------------------------------------------------------------------
# Calculate improved Is with transmission/reflection losses
Is = np.zeros((steps, num_freqs))
for j, f in enumerate(frequencies):
    for i, d in enumerate(z_cm):
        intensity = I0
        remaining = d
        for l, mat in enumerate(material_list):
            if remaining <= 0:
                break
            layer_len = min(mat.Thickness, remaining)

            # Attenuation
            total_attn = mat.MuA * f * layer_len
            intensity *= 10 ** (-total_attn / 10)

            # Transmission loss at interface (not at last layer)
            if l < len(material_list) - 1 and remaining > layer_len:
                next_mat = material_list[l + 1]
                T = transmission(mat, next_mat)
                intensity *= T

            remaining -= layer_len

        Is[i, j] = intensity  # Store improved intensity

# SNR at max depth, with speckle + noise
MC_REPS = 1000
electronic_noise_std = 0.001  # W/cm^2 from assignment
SNR_maxdepth = np.zeros(num_freqs)

for j, f in enumerate(frequencies):
    signals = []
    for _ in range(MC_REPS):
        # Multiplicative speckle (Rayleigh)
        speckle = np.random.rayleigh(scale=1/np.sqrt(np.pi/2))
        # Additive noise
        noise = np.random.normal(0, electronic_noise_std)
        measured = Is[-1, j] * speckle + noise
        signals.append(measured)
    signals = np.array(signals)
    SNR_maxdepth[j] = np.abs(np.mean(signals)) / np.std(signals)

SNR_db_maxdepth = 20 * np.log10(SNR_maxdepth)

# Plot result
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(frequencies, SNR_db_maxdepth, marker="o")
plt.xlabel("Frequency (MHz)")
plt.ylabel("SNR at Max Depth (dB)")
plt.title("SNR at Maximum Depth vs Frequency\nWith Transmission & Reflection Losses")
plt.grid(True)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------------
# Monte Carlo simulation: SNR at each (depth, freq)
# ------------------------------------------------------------------
SNR = np.zeros((steps, num_freqs))
for j, f in enumerate(frequencies):
    for i in range(steps):
        signals = []
        for _ in range(MC_REPS):
            # Multiplicative speckle
            speckle = np.random.rayleigh(scale=1/np.sqrt(np.pi/2))  # mean ≈1
            # Additive electronic noise
            noise   = np.random.normal(0, electronic_noise_std)
            measured = Is[i, j] * speckle + noise
            signals.append(measured)
        signals = np.array(signals)
        SNR[i, j] = np.abs(np.mean(signals)) / np.std(signals)  # abs prevents mean flip at low SNR

SNR_db = 20 * np.log10(SNR)
SNR_db[np.isneginf(SNR_db)|np.isnan(SNR_db)] = np.nan  # Clean up -inf/nan for plotting

# ------------------------------------------------------------------
# Plot: Intensity vs. Depth (each frequency)
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
for j, f in enumerate(frequencies):
    plt.plot(z_mm, Is[:, j], label=f"{f:.0f} MHz")
plt.xlabel("Depth (mm)")
plt.ylabel("Simulated Backscatter Intensity (W/cm$^2$)")
plt.title("Backscatter Intensity vs Depth")
plt.legend()
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Plot: SNR (dB) vs. Frequency at max depth
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(frequencies, SNR_db[-1, :], marker='o')
plt.xlabel("Frequency (MHz)")
plt.ylabel("SNR at Max Depth (dB)")
plt.title("SNR vs Frequency at Maximum Depth")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Plot: SNR (dB) vs. Depth for highest frequency
# ------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(z_mm, SNR_db[:, -1], label=f"{frequencies[-1]:.0f} MHz")
plt.xlabel("Depth (mm)")
plt.ylabel("SNR (dB)")
plt.title(f"SNR vs Depth at {frequencies[-1]:.0f} MHz")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# Print summary SNR table at key depths
# ------------------------------------------------------------------
print("Depth (mm)   """"  """ + "   ".join([f"{f:.0f} MHz" for f in frequencies]))
for i in np.linspace(0, steps-1, 5, dtype=int):
    row = SNR_db[i]
    print(f"{z_mm[i]:6.1f}     " + "   ".join([f"{v:6.1f}" for v in row]))


tumor_nerve_index = np.argmax(z_cm > (Fat.Thickness + Muscle.Thickness + Tumor.Thickness))
for j, f in enumerate(frequencies):
    nerve_signal = Is[tumor_nerve_index, j]
    tumor_signal = Is[tumor_nerve_index-1, j]
    cnr = abs(nerve_signal - tumor_signal) / np.sqrt(nerve_signal + tumor_signal)
    print(f"CNR at tumor/nerve for {f} MHz: {cnr:.5f}")



