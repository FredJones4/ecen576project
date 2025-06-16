#!/usr/bin/env python3
# realistic_main.py
# Option 1: Supply explicit ICRU-44 weight‐fraction formulas for each tissue.

import numpy as np
import matplotlib.pyplot as plt
import xraydb as xdb  # pip install xraydb>=4.5.6

# -------------------------------------------------------------
# Tissue properties
# -------------------------------------------------------------
tissues = {
    "fat": {
        "thickness_cm": 0.5,
        "density_g_per_cm3": 0.92,
        "T1_ms": 337,
        "T2_ms": 98,
    },
    "muscle": {
        "thickness_cm": 3.0,
        "density_g_per_cm3": 1.05,
        "T1_ms": 1233,
        "T2_ms": 37,
    },
    "tumor": {
        "thickness_cm": 1.0,
        "density_g_per_cm3": 1.06,
        "T1_ms": 1100,
        "T2_ms": 50,
    },
}

modalities = {
    "MRI": {"SNR_ref": 20.0},
    "Ultrasound": {"electronic_noise_std_W_cm2": 0.001},
    "Xray_CT": {
        "kVp": 120.0,
        "filter_total_Al_cm": 0.25 + 0.60,  # inherent + bow-tie
        "detector_efficiency": 0.80,
        "baseline_electronic_noise_HU": 5.0,
    },
}

# -------------------------------------------------------------
# Energy grid and spectrum generator
# -------------------------------------------------------------
def tungsten_brems_spectrum(E_keV, kVp):
    spec = np.maximum(kVp - E_keV, 0) * E_keV
    return spec / spec.max()

energies_keV = np.arange(20, modalities["Xray_CT"]["kVp"] + 1)
energies_eV = energies_keV * 1e3

# -------------------------------------------------------------
# X-ray filtration
# -------------------------------------------------------------
RHO_AL = 2.6989  # g/cm³
mu_rho_Al = np.array([xdb.mu_elam("Al", E) for E in energies_eV])
mu_Al = mu_rho_Al * RHO_AL
filter_T = np.exp(-mu_Al * modalities["Xray_CT"]["filter_total_Al_cm"])

S0 = tungsten_brems_spectrum(energies_keV, modalities["Xray_CT"]["kVp"])
S_filtered = S0 * filter_T
dE = 1.0

# -------------------------------------------------------------
# Helper: µ(E) for arbitrary ICRU-44 formula
# -------------------------------------------------------------
ICRU44_FORMULAS = {
    "muscle": (
        "H0.103C0.143N0.034O0.710"
        "Na0.0008P0.0008S0.0020Cl0.0008Ca0.0007"
    ),
    "tumor": (
        # Example: same base as muscle, but tweak if desired
        "H0.103C0.143N0.034O0.710"
        "Na0.0008P0.0008S0.0020Cl0.0008Ca0.0007"
    ),
}

def mu_tissue_cm1(tissue_key: str) -> np.ndarray:
    rho = tissues[tissue_key]["density_g_per_cm3"]
    formula = ICRU44_FORMULAS[tissue_key]
    return np.array([
        xdb.material_mu(formula, E, density=rho)
        for E in energies_eV
    ])

# -------------------------------------------------------------
# Compute transmitted signals and CNR
# -------------------------------------------------------------
mu_muscle_E = mu_tissue_cm1("muscle")
mu_tumor_E  = mu_tissue_cm1("tumor")

def transmitted(mu_E, thickness_cm):
    return np.sum(S_filtered * np.exp(-mu_E * thickness_cm)) * dE

I_muscle = transmitted(mu_muscle_E, tissues["muscle"]["thickness_cm"])
I_tumor  = transmitted(mu_muscle_E, tissues["muscle"]["thickness_cm"]) * \
           np.exp(-np.sum(mu_tumor_E * tissues["tumor"]["thickness_cm"]))

η = modalities["Xray_CT"]["detector_efficiency"]
I_muscle_det, I_tumor_det = I_muscle * η, I_tumor * η
σ_quantum = np.sqrt(I_muscle_det)
HU_per_cnt = 1000 / I_muscle_det
σ_elec = modalities["Xray_CT"]["baseline_electronic_noise_HU"] / HU_per_cnt
σ_total = np.sqrt(σ_quantum**2 + σ_elec**2)
CNR_CT = abs(I_tumor_det - I_muscle_det) / σ_total

# -------------------------------------------------------------
# Plot and report
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(energies_keV, S0, label="Unfiltered")
plt.plot(
    energies_keV,
    S_filtered,
    label=f"After {modalities['Xray_CT']['filter_total_Al_cm']*10:.1f} mm Al",
)
plt.xlabel("Energy (keV)")
plt.ylabel("Relative fluence (a.u.)")
plt.title("120 kVp spectrum – realistic filtration")
plt.legend()
plt.tight_layout()
plt.show()

print(f"▶ Realistic CT tumour/muscle CNR ≈ {CNR_CT:.2f}")
