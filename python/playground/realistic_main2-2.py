#!/usr/bin/env python3
# realistic_main_named.py
# Option 2 (fixed): Use xdb.get_materials() to list built-in materials.
# ERROR: Named material 'ICRU-44 Muscle' not found in xdb.get_materials().
import numpy as np
import matplotlib.pyplot as plt
import xraydb as xdb  # pip install xraydb>=4.5.6

# -------------------------------------------------------------
# Tissue & modality settings (same as before)
# -------------------------------------------------------------
tissues = {
    "muscle": {"thickness_cm": 3.0, "density_g_per_cm3": 1.05},
    "tumor":  {"thickness_cm": 1.0, "density_g_per_cm3": 1.06},
}

modalities = {
    "Xray_CT": {
        "kVp": 120.0,
        "filter_total_Al_cm": 0.25 + 0.60,
        "detector_efficiency": 0.80,
        "baseline_electronic_noise_HU": 5.0,
    },
}

# -------------------------------------------------------------
# Spectrum & filtration (unchanged)
# -------------------------------------------------------------
def tungsten_brems_spectrum(E_keV, kVp): # Assuming constant tube current, that we only care about proportionality, and combining equations (1) and (3) from https://floban.folk.ntnu.no/KJ%20%203055/X%20%20Ray/Bremsstrahlung.htm
    spec = np.maximum(kVp - E_keV, 0) * E_keV
    return spec / spec.max()

energies_keV = np.arange(20, modalities["Xray_CT"]["kVp"] + 1)
energies_eV  = energies_keV * 1e3

# Aluminium filter attenuation
RHO_AL    = 2.6989
mu_rho_Al = np.array([xdb.mu_elam("Al", E) for E in energies_eV]) # array of 'absorption cross-section, photo-electric or total for an element'; https://xraypy.github.io/XrayDB/xraydb.pdf
mu_Al     = mu_rho_Al * RHO_AL
T_filter  = np.exp(-mu_Al * modalities["Xray_CT"]["filter_total_Al_cm"])
S0        = tungsten_brems_spectrum(energies_keV, modalities["Xray_CT"]["kVp"])
S_filtered= S0 * T_filter
dE        = 1.0

# -------------------------------------------------------------
# Fixed material lookup via get_materials()
# -------------------------------------------------------------
all_materials = xdb.get_materials()  # dict of {name: (formula, density, categories)} :contentReference[oaicite:3]{index=3}
available     = {name.lower() for name in all_materials.keys()}

if "icru-44 muscle" not in available:
    raise RuntimeError(
        "Named material 'ICRU-44 Muscle' not found in xdb.get_materials()."
    )

def mu_tissue_cm1_named(tissue_key: str) -> np.ndarray:
    """µ(E) [cm⁻¹] via built-in 'ICRU-44 Muscle' entry."""
    rho = tissues[tissue_key]["density_g_per_cm3"]
    # use the same named material for muscle & tumor
    mat_name = next(                                    # Efficient iteration method for grabbing the first item that meets the condition on the list
        name for name in all_materials.keys()
        if name.lower() == "icru-44 muscle"
    )
    return np.array([
        xdb.material_mu(mat_name, E, density=rho)
        for E in energies_eV
    ])

# -------------------------------------------------------------
# Compute transmitted signals & CNR (same as before)
# -------------------------------------------------------------
mu_muscle_E = mu_tissue_cm1_named("muscle")
mu_tumor_E  = mu_tissue_cm1_named("tumor")

def transmitted(mu_E, thickness_cm):
    return np.sum(S_filtered * np.exp(-mu_E * thickness_cm)) * dE

I_muscle = transmitted(mu_muscle_E, tissues["muscle"]["thickness_cm"])
I_tumor  = transmitted(mu_muscle_E, tissues["muscle"]["thickness_cm"]) * \
           np.exp(-np.sum(mu_tumor_E * tissues["tumor"]["thickness_cm"]))

η            = modalities["Xray_CT"]["detector_efficiency"]
I_muscle_det  = I_muscle * η
I_tumor_det   = I_tumor  * η
σ_quantum    = np.sqrt(I_muscle_det)
HU_per_cnt   = 1000 / I_muscle_det
σ_elec       = modalities["Xray_CT"]["baseline_electronic_noise_HU"] / HU_per_cnt
σ_total      = np.sqrt(σ_quantum**2 + σ_elec**2)
CNR_CT       = abs(I_tumor_det - I_muscle_det) / σ_total

# -------------------------------------------------------------
# Plot & report
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.plot(energies_keV, S0,          label="Unfiltered")
plt.plot(energies_keV, S_filtered,  label=f"After {modalities['Xray_CT']['filter_total_Al_cm']*10:.1f} mm Al")
plt.xlabel("Energy (keV)")
plt.ylabel("Relative fluence (a.u.)")
plt.title("120 kVp spectrum – realistic filtration")
plt.legend(); plt.tight_layout(); plt.show()

print(f"▶ Realistic CT tumour/muscle CNR ≈ {CNR_CT:.2f}")
