"""
CNR optimisation & noise demonstrations
Powered by AbracaTABra tabs
Requires:
    pip install "abracatabra[qt-pyside6]"   # Qt backend + helper
"""

import numpy as np
import matplotlib.pyplot as plt
import abracatabra                    # ⭐ NEW

# -------------------------------------------------
# 0. Tissue & modality dictionaries
#     (identical to your earlier snippet)
# -------------------------------------------------
tissues = {
    "fat":    {"thickness_cm": 0.5, "shape": "cylinder shell", "density_g_per_cm3": 0.6,
               "acoustic_absorption_db_cm_mhz": 0.9, "acoustic_speed_m_per_s": 1476,
               "xray_mu_linear": -0.0004, "xray_mu_intercept": 0.196, "T1_ms": 337, "T2_ms": 98},
    "muscle": {"thickness_cm": 3.0, "shape": "cylinder shell", "density_g_per_cm3": 0.9,
               "acoustic_absorption_db_cm_mhz": 0.54, "acoustic_speed_m_per_s": 1580,
               "xray_mu_linear": -0.0065, "xray_mu_intercept": 0.26,  "T1_ms": 1233, "T2_ms": 37},
    "tumor":  {"thickness_cm": 1.0, "shape": "sphere",          "density_g_per_cm3": 0.8,
               "acoustic_absorption_db_cm_mhz": 0.76, "acoustic_speed_m_per_s": 1564,
               "xray_mu_linear": -0.0008, "xray_mu_intercept": 0.25,  "T1_ms": 1100, "T2_ms": 50},
    "nerve":  {"thickness_cm": 0.5, "shape": "cylinder",        "density_g_per_cm3": 0.9,
               "acoustic_absorption_db_cm_mhz": 0.9,  "acoustic_speed_m_per_s": 1630,
               "xray_mu_linear": -0.0065, "xray_mu_intercept": 0.24,  "T1_ms": 1083, "T2_ms": 78},
}

modalities = {
    "MRI":        {"voxel_volume_mm3": 1.0, "SNR_ref": 20.0},
    "Ultrasound": {"input_intensity_W_cm2": 0.1, "electronic_noise_std_W_cm2": 0.001}, # not listed: speckle is multiplicative noise w/ Rayleigh Distribution w/ mu=1, array size and frequency can be changed, assume linear array to avoid refraction.
    "Xray_CT":    {"noise_distribution": "Poisson", "tube_voltage_kVp_max": 120}, # not listed: Max current 300 A, 1% conversion efficiency at anode, detector is 100% efficient.
}

# -------------------------------------------------
# 1. MRI: global optimum CNR on a TR–TE grid
# -------------------------------------------------
TR_vals_ms = np.linspace(200, 3000, 60)
TE_vals_ms = np.linspace(5, 120, 60)
TE_grid, TR_grid = np.meshgrid(TE_vals_ms, TR_vals_ms)

def mri_signal(T1, T2, TR, TE, S0=1.0):
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

S_muscle = mri_signal(tissues["muscle"]["T1_ms"], tissues["muscle"]["T2_ms"],
                      TR_grid, TE_grid)
S_tumor  = mri_signal(tissues["tumor"]["T1_ms"],  tissues["tumor"]["T2_ms"],
                      TR_grid, TE_grid)
S_nerve = mri_signal(tissues["nerve"]["T1_ms"],  tissues["tumor"]["T2_ms"],
                      TR_grid, TE_grid)

sigma_ref   = S_muscle.max() / modalities["MRI"]["SNR_ref"]
CNR_grid    = np.abs(S_tumor - S_muscle) / sigma_ref
imax        = np.unravel_index(np.argmax(CNR_grid), CNR_grid.shape)
TR_opt, TE_opt, CNR_opt = TR_vals_ms[imax[0]], TE_vals_ms[imax[1]], CNR_grid[imax]

# -------------------------------------------------
# 2. Ultrasound: depth-wise intensity with speckle
# -------------------------------------------------
freq_MHz  = 5.0
depths_cm = np.linspace(0,
                        tissues["fat"]["thickness_cm"]
                        + tissues["muscle"]["thickness_cm"]
                        + tissues["tumor"]["thickness_cm"],
                        300)

def cumulative_intensity(depth_cm):
    I = 1.0
    remain = depth_cm
    for layer in ["fat", "muscle", "tumor"]:
        t_layer = tissues[layer]["thickness_cm"]
        if remain <= 0:
            break
        step = min(t_layer, remain)
        mu_db_per_cm = tissues[layer]["acoustic_absorption_db_cm_mhz"] * freq_MHz
        I *= 10 ** (-mu_db_per_cm * step / 10)
        remain -= step
    return I

I_depth = np.array([cumulative_intensity(d) for d in depths_cm])

sigma_e       = modalities["Ultrasound"]["electronic_noise_std_W_cm2"]
cv_rayleigh   = np.sqrt((4 - np.pi) / np.pi)          # ≈0.522
sigma_speckle = cv_rayleigh * I_depth
sigma_total   = np.sqrt(sigma_e**2 + sigma_speckle**2)

idx_boundary     = np.searchsorted(depths_cm,
                                   tissues["fat"]["thickness_cm"]
                                   + tissues["muscle"]["thickness_cm"])
I_muscle_layer   = I_depth[idx_boundary - 1]
I_tumor_surface  = I_depth[idx_boundary]
sigma_muscle     = sigma_total[idx_boundary - 1]
CNR_ultrasound   = np.abs(I_tumor_surface - I_muscle_layer) / sigma_muscle

# -------------------------------------------------
# 3. X-ray: polychromatic spectrum & bow-tie filter
# -------------------------------------------------
kVp          = 120.0
energies_keV = np.linspace(20, kVp, 400)

def tungsten_brems_spectrum(E, kVp):
    return np.maximum(kVp - E, 0) * E                  # ∝ E·(kVp−E)

def mu_aluminum(E_keV):
    return 5e-3 * (30 / E_keV)**3                      # crude E⁻³ fall-off

S_unfiltered = tungsten_brems_spectrum(energies_keV, kVp)
S_filtered   = S_unfiltered * np.exp(-mu_aluminum(energies_keV) * 1.0)  # 1 cm Al

def mu_tissue(tissue, E_keV):
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E_keV + b

t_muscle, t_tumor = tissues["muscle"]["thickness_cm"], tissues["tumor"]["thickness_cm"]
dE = energies_keV[1] - energies_keV[0]

def transmitted(path_func):
    return np.sum(S_filtered * np.exp(-path_func(energies_keV))) * dE

I_muscle_poly = transmitted(lambda E: mu_tissue("muscle", E) * t_muscle)
I_tumor_poly  = transmitted(lambda E: mu_tissue("muscle", E) * t_muscle
                                      + mu_tissue("tumor",  E) * t_tumor)
CNR_xray_poly = np.abs(I_tumor_poly - I_muscle_poly) / np.sqrt(I_muscle_poly)

# =================================================
#     🪄  BUILD ONE TAB-ORGANIZED PLOT WINDOW
# =================================================
window = abracatabra.TabbedPlotWindow(window_id="Multi-modal-CNR", ncols=2)

# ---- (1) MRI CNR heat-map ---------------------------------------
fig = window.add_figure_tab("MRI CNR map", col=0)
ax  = fig.add_subplot()
im  = ax.imshow(CNR_grid,
                extent=[TE_vals_ms.min(), TE_vals_ms.max(),
                        TR_vals_ms.max(), TR_vals_ms.min()],
                aspect='auto')
ax.scatter(TE_opt, TR_opt, marker='x', s=80, color='white')
ax.set(xlabel="TE (ms)", ylabel="TR (ms)",
       title=f"MRI CNR vs TR & TE\nopt: TR≈{TR_opt:.0f} ms, TE≈{TE_opt:.0f} ms\nCNR≈{CNR_opt:.1f}")
fig.colorbar(im, ax=ax, label="CNR (Tumour – Muscle)")

# ---- (2) Ultrasound depth profile -------------------------------
fig = window.add_figure_tab("Ultrasound depth", col=1)
ax  = fig.add_subplot()
ax.plot(depths_cm, I_depth, label="Mean intensity")
ax.fill_between(depths_cm, I_depth - sigma_speckle, I_depth + sigma_speckle,
                alpha=0.3, label="±σ (speckle)")
ax.set(xlabel="Depth (cm)", ylabel="Intensity (norm.)",
       title=f"Ultrasound 5 MHz – CNR≈{CNR_ultrasound:.2f}")
ax.legend()
ax.grid(True)

# ---- (3) X-ray spectra -----------------------------------------
fig = window.add_figure_tab("X-ray spectrum")
ax  = fig.add_subplot()
ax.plot(energies_keV, S_unfiltered / S_unfiltered.max(), label="Unfiltered")
ax.plot(energies_keV, S_filtered   / S_filtered.max(),   label="After 1 cm Al")
ax.set(xlabel="Energy (keV)", ylabel="Relative intensity",
       title=f"120 kVp, bow-tie filtered – CNR≈{CNR_xray_poly:.2f}")
ax.legend()
ax.grid(True)

window.apply_tight_layout()
abracatabra.abracatabra()               # ⭐ replaces plt.show()

# -------------------------------------------------
# Console summary (unchanged)
# -------------------------------------------------
print(f"\n▶ MRI optimum: TR ≈ {TR_opt:.0f} ms, TE ≈ {TE_opt:.0f} ms, CNR ≈ {CNR_opt:.1f}")
print(f"▶ Ultrasound tumour/muscle CNR with speckle noise: {CNR_ultrasound:.2f}")
print(f"▶ X-ray tumour/muscle CNR (poly, filtered): {CNR_xray_poly:.2f}")
