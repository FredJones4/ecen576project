import numpy as np
import matplotlib.pyplot as plt

# Re-use dictionaries from the prior run – they are still in memory.
# (If you restarted your kernel, just re-run the earlier block before executing this one.)
# --------------------------
# 1. Dictionary of terms
# --------------------------
tissues = {
    "fat": {
        "thickness_cm": 0.5,
        "shape": "cylinder shell",
        "density_g_per_cm3": 0.6,
        "acoustic_absorption_db_cm_mhz": 0.9,
        "acoustic_speed_m_per_s": 1476,
        "xray_mu_linear": -0.0004,  # slope m in cm^-1 keV^-1
        "xray_mu_intercept": 0.196,  # intercept b
        "T1_ms": 337,
        "T2_ms": 98,
    },
    "muscle": {
        "thickness_cm": 3.0,
        "shape": "cylinder shell",
        "density_g_per_cm3": 0.9,
        "acoustic_absorption_db_cm_mhz": 0.54,
        "acoustic_speed_m_per_s": 1580,
        "xray_mu_linear": -0.0065,
        "xray_mu_intercept": 0.26,
        "T1_ms": 1233,
        "T2_ms": 37,
    },
    "tumor": {
        "thickness_cm": 1.0,
        "shape": "sphere",
        "density_g_per_cm3": 0.8,
        "acoustic_absorption_db_cm_mhz": 0.76,
        "acoustic_speed_m_per_s": 1564,
        "xray_mu_linear": -0.0008,
        "xray_mu_intercept": 0.25,
        "T1_ms": 1100,
        "T2_ms": 50,
    },
    "nerve": {
        "thickness_cm": 0.5,  # up to 0.5 mm below tumor, take total segment
        "shape": "cylinder",
        "density_g_per_cm3": 0.9,
        "acoustic_absorption_db_cm_mhz": 0.9,
        "acoustic_speed_m_per_s": 1630,
        "xray_mu_linear": -0.0065,
        "xray_mu_intercept": 0.24,
        "T1_ms": 1083,
        "T2_ms": 78,
    },
}

modalities = {
    "MRI": {
        "voxel_volume_mm3": 1.0,
        "SNR_ref": 20.0,  # SNR for 1 mm^3 voxel, infinite TR, TE≈0
        "noise_distribution": "Rician",
        "min_voxel_mm": 0.25,
        "typical_params": ["TR (ms)", "TE (ms)", "Resolution", "N_TR"],
    },
    "Ultrasound": {
        "input_intensity_W_cm2": 0.1,
        "electronic_noise_std_W_cm2": 0.001,
        "noise_distribution": "Gaussian additive + Rayleigh multiplicative (speckle)",
        "assumptions": "Linear array, adjustable carrier frequency & array width",
        "typical_params": ["Array width", "Carrier frequency (MHz)", "Axial/Lateral resolution"],
    },
    "Xray_CT": {
        "noise_distribution": "Poisson (quantum) limited",
        "tube_voltage_kVp_max": 120,
        "tube_current_A_max": 300,
        "conversion_efficiency": 0.01,
        "detector_efficiency": 1.0,
        "typical_params": ["kVp", "Current (A)", "Exposure time (s)", "Resolution"],
    },
}

# ------------------------------------------------------------------
# 1. MRI · Global optimum CNR across TR–TE grid
# ------------------------------------------------------------------
TR_vals_ms = np.linspace(200, 3000, 60)      # 60 points from 0.2-3 s
TE_vals_ms = np.linspace(5, 120, 60)         # 60 points 5-120 ms
TE_grid, TR_grid = np.meshgrid(TE_vals_ms, TR_vals_ms)

def mri_signal(T1, T2, TR, TE, S0=1.0):
    """Spin-echo signal equation."""
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

# Muscle & tumour signals on the full grid
S_muscle = mri_signal(
    tissues["muscle"]["T1_ms"],
    tissues["muscle"]["T2_ms"],
    TR_grid, TE_grid
)
S_tumor = mri_signal(
    tissues["tumor"]["T1_ms"],
    tissues["tumor"]["T2_ms"],
    TR_grid, TE_grid
)

# Noise σ chosen as muscle signal at TE→0, TR large, divided by SNR_ref
sigma_ref = S_muscle.max() / modalities["MRI"]["SNR_ref"]

CNR_grid = np.abs(S_tumor - S_muscle) / sigma_ref
imax = np.unravel_index(np.argmax(CNR_grid), CNR_grid.shape)
TR_opt, TE_opt, CNR_opt = TR_vals_ms[imax[0]], TE_vals_ms[imax[1]], CNR_grid[imax]

# Plot heat-map
plt.figure()
plt.imshow(CNR_grid, extent=[TE_vals_ms.min(), TE_vals_ms.max(),
                             TR_vals_ms.max(), TR_vals_ms.min()],
           aspect='auto')
plt.scatter(TE_opt, TR_opt, marker='x', s=80)  # optimum
plt.colorbar(label="CNR (Tumour – Muscle)")
plt.xlabel("TE (ms)")
plt.ylabel("TR (ms)")
plt.title("MRI CNR vs TR & TE\noptimum: TR≈%.0f ms, TE≈%.0f ms, CNR≈%.1f"
          % (TR_opt, TE_opt, CNR_opt))
plt.grid(False)

# ------------------------------------------------------------------
# 2. Ultrasound · add Rayleigh speckle noise (multiplicative)
# ------------------------------------------------------------------
freq_MHz = 5.0
depths_cm = np.linspace(0,
                        tissues["fat"]["thickness_cm"] +
                        tissues["muscle"]["thickness_cm"] +
                        tissues["tumor"]["thickness_cm"], 300)

def cumulative_intensity(depth):
    I = 1.0
    remaining = depth
    for layer in ["fat", "muscle", "tumor"]:
        t = tissues[layer]["thickness_cm"]
        if remaining <= 0:
            break
        traversed = min(t, remaining)
        mu_db_per_cm = tissues[layer]["acoustic_absorption_db_cm_mhz"] * freq_MHz
        I *= 10 ** (-mu_db_per_cm * traversed / 10)
        remaining -= traversed
    return I

I_depth = np.array([cumulative_intensity(d) for d in depths_cm])

# Noise terms
sigma_e = modalities["Ultrasound"]["electronic_noise_std_W_cm2"]
cv_rayleigh = np.sqrt((4 - np.pi) / np.pi)   # ≈0.522
sigma_speckle = cv_rayleigh * I_depth
sigma_total = np.sqrt(sigma_e**2 + sigma_speckle**2)

# Tumour surface vs muscle layer just above it
idx_boundary = np.searchsorted(depths_cm,
                               tissues["fat"]["thickness_cm"] +
                               tissues["muscle"]["thickness_cm"])
I_muscle_layer = I_depth[idx_boundary - 1]
I_tumor_surface = I_depth[idx_boundary]
sigma_muscle = sigma_total[idx_boundary - 1]

CNR_ultrasound = np.abs(I_tumor_surface - I_muscle_layer) / sigma_muscle

# Plot intensity with speckle noise band
plt.figure()
plt.plot(depths_cm, I_depth, label="Mean intensity")
plt.fill_between(depths_cm,
                 I_depth - sigma_speckle,
                 I_depth + sigma_speckle,
                 alpha=0.3, label="±σ (speckle)")
plt.xlabel("Depth (cm)")
plt.ylabel("Intensity (norm.)")
plt.title("Ultrasound Intensity vs Depth\n5 MHz, speckle included\nCNR(Tumour/Muscle)≈%.2f"
          % CNR_ultrasound)
plt.legend()
plt.grid(True)

# ------------------------------------------------------------------
# 3. X-ray / CT · polychromatic beam & bow-tie filtration
# ------------------------------------------------------------------
kVp = 120.0
energies_keV = np.linspace(20, kVp, 400)

def tungsten_brems_spectrum(E, kVp):
    """Simple unfiltered Bremsstrahlung shape ∝ E·(kVp-E)."""
    s = np.maximum(kVp - E, 0) * E
    return s

def mu_aluminum(E):
    """Crude μ(E) for Al in cm-1; scales ~E⁻³ (not for quantitative dose!)."""
    return 5e-3 * (30 / E)**3

# Unfiltered & filtered spectra
S_unfiltered = tungsten_brems_spectrum(energies_keV, kVp)
filter_thickness_cm = 1.0  # equivalent Al thickness
S_filtered = S_unfiltered * np.exp(-mu_aluminum(energies_keV) * filter_thickness_cm)

# Tissue attenuation coefficients
def mu_tissue(tissue, E):
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E + b

# Path lengths
t_muscle = tissues["muscle"]["thickness_cm"]
t_tumor = tissues["tumor"]["thickness_cm"]

# Polychromatic transmitted signals
dE = energies_keV[1] - energies_keV[0]
def transmitted_intensity(path_func):
    """Integrate S_filtered * exp(-Σ μ_i(E)·t_i) dE."""
    att = np.exp(-path_func(energies_keV))
    return np.sum(S_filtered * att) * dE

I_muscle_poly = transmitted_intensity(lambda E:
                                      mu_tissue("muscle", E) * t_muscle)
I_tumor_poly = transmitted_intensity(lambda E:
                                     mu_tissue("muscle", E) * t_muscle +
                                     mu_tissue("tumor", E) * t_tumor)

CNR_xray_poly = np.abs(I_tumor_poly - I_muscle_poly) / np.sqrt(I_muscle_poly)

# Plot spectra
plt.figure()
plt.plot(energies_keV, S_unfiltered / S_unfiltered.max(), label="Unfiltered")
plt.plot(energies_keV, S_filtered / S_filtered.max(), label="After 1 cm Al bow-tie")
plt.xlabel("Energy (keV)")
plt.ylabel("Relative spectral intensity")
plt.title("X-ray Spectrum – 120 kVp\nPolychromatic, bow-tie filtered\nCNR(Tumour/Muscle)≈%.2f"
          % CNR_xray_poly)
plt.legend()
plt.grid(True)

plt.show()

# Print key numeric results
print(f"\n▶ MRI optimum: TR ≈ {TR_opt:.0f} ms, TE ≈ {TE_opt:.0f} ms, CNR ≈ {CNR_opt:.1f}")
print(f"▶ Ultrasound tumour/muscle CNR with speckle noise: {CNR_ultrasound:.2f}")
print(f"▶ X-ray tumour/muscle CNR (poly, filtered): {CNR_xray_poly:.2f}")
