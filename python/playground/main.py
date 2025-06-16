import numpy as np
import matplotlib.pyplot as plt

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

# Display the dictionaries to the user
print("Tissue properties dictionary:")
for k, v in tissues.items():
    print(f"  {k}: {v}")
print("\nModality assumptions dictionary:")
for k, v in modalities.items():
    print(f"  {k}: {v}")

# --------------------------
# 2. MRI Contrast Calculation
# --------------------------
TR_ms = 500  # repetition time (ms)
TE_values_ms = np.linspace(5, 100, 40)  # echo times for plotting

def mri_signal(T1, T2, TR, TE, S0=1.0): # Normalized S0
    """Return relative signal magnitude for spin-echo."""
    return S0 * (1 - np.exp(-TR/T1)) * np.exp(-TE/T2) # https://radiopaedia.org/articles/spin-echo-sequences  || https://www.cis.rit.edu/htbooks/mri/chap-10/chap-10.htm#:~:text=The%20signal%20equations%20for%20the,Inversion%20Recovery%20(180%2D90)


signals = {t: mri_signal(
                tissues[t]["T1_ms"],
                tissues[t]["T2_ms"],
                TR_ms,
                TE_values_ms
           )
           for t in ["fat", "muscle", "tumor", "nerve"]}

# Contrast-to-noise ratio (CNR) between tumor and muscle
# Assume noise σ_ref gives SNR_ref=20 at TE→0 for muscle
print(signals["muscle"]) # Show that AI exploited Python's ability to run an array of values from TE_values in the one function.
sigma_noise = signals["muscle"][0] / modalities["MRI"]["SNR_ref"] # page 77 of textbook, solve for σ_ref based on first value
cnr_tumor_muscle = np.abs(signals["tumor"] - signals["muscle"]) / sigma_noise # This is an array of values; see also HW 9

# Plot MRI signal vs TE
plt.figure()
for t, s in signals.items():
    plt.plot(TE_values_ms, s, label=t.capitalize())
plt.xlabel("Echo Time TE (ms)")
plt.ylabel("Relative MRI Signal")
plt.title("Spin-Echo MRI Signal vs TE (TR = 500 ms)")
plt.legend()
plt.grid(True)

# Plot CNR
plt.figure()
plt.plot(TE_values_ms, cnr_tumor_muscle)
plt.xlabel("Echo Time TE (ms)")
plt.ylabel("CNR (Tumor vs Muscle)")
plt.title("MRI CNR Tumor–Muscle vs TE")
plt.grid(True)

# --------------------------
# 3. X-ray Contrast Calculation
# --------------------------
energies_keV = np.linspace(20, 120, 200)  # diagnostic range
thickness_cm_path = tissues["muscle"]["thickness_cm"] + tissues["tumor"]["thickness_cm"]

def linear_attenuation(tissue, E):
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E + b

I0 = 1.0  # normalized incident intensity

I_muscle = I0 * np.exp(-linear_attenuation("muscle", energies_keV) *
                       tissues["muscle"]["thickness_cm"])
I_tumor = I0 * np.exp(-(linear_attenuation("tumor", energies_keV) *
                       tissues["tumor"]["thickness_cm"] +
                       linear_attenuation("muscle", energies_keV) *
                       tissues["muscle"]["thickness_cm"]))  # tumor behind muscle

# Poisson noise std = sqrt(I)
snr_muscle = I_muscle / np.sqrt(I_muscle)
cnr_xray = np.abs(I_tumor - I_muscle) / np.sqrt(I_muscle)

# Plot transmitted intensity
plt.figure()
plt.plot(energies_keV, I_muscle, label="Muscle path")
plt.plot(energies_keV, I_tumor, label="Muscle+Tumor path")
plt.xlabel("X-ray Energy (keV)")
plt.ylabel("Transmitted Intensity (norm.)")
plt.title("X-ray Transmission vs Energy")
plt.legend()
plt.grid(True)

# Plot CNR
plt.figure()
plt.plot(energies_keV, cnr_xray)
plt.xlabel("X-ray Energy (keV)")
plt.ylabel("CNR (Tumor vs Muscle)")
plt.title("X-ray CNR Tumor–Muscle vs Energy")
plt.grid(True)

# --------------------------
# 4. Ultrasound Contrast Calculation
# --------------------------
freq_MHz = 5.0
depths_cm = np.linspace(0, tissues["fat"]["thickness_cm"] + tissues["muscle"]["thickness_cm"] +
                           tissues["tumor"]["thickness_cm"], 300)

def cumulative_intensity(depth):
    """Compute intensity at a given depth accounting for layers."""
    I = I0
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
# Electronic noise σ_e
sigma_e = modalities["Ultrasound"]["electronic_noise_std_W_cm2"]
# CNR between tumor and muscle boundary (simplified)
boundary_index = np.searchsorted(depths_cm,
                                 tissues["fat"]["thickness_cm"] + tissues["muscle"]["thickness_cm"])
I_muscle_layer = I_depth[boundary_index - 1]
I_tumor_surface = I_depth[boundary_index]
cnr_ultrasound = np.abs(I_tumor_surface - I_muscle_layer) / sigma_e

# Plot depth intensity
plt.figure()
plt.plot(depths_cm, I_depth)
plt.xlabel("Depth (cm)")
plt.ylabel("Intensity (norm.)")
plt.title("Ultrasound Intensity vs Depth (5 MHz)")
plt.grid(True)

plt.show()
