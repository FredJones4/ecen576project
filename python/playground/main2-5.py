"""
CNR optimisation & noise demonstrations – extended
Powered by AbracaTABra tabs

Requires:
    pip install "abracatabra[qt-pyside6]"
"""

import numpy as np
import matplotlib.pyplot as plt
import abracatabra                              # ⭐

# -------------------------------------------------
# 0. Tissue & modality dictionaries (unchanged)
# -------------------------------------------------
tissues = {
    "fat":    {"thickness_cm": 0.5, "density": 0.6,
               "α_db_cm_mhz": 0.9,  "c_m_per_s": 1476,
               "m_mu": -0.0004, "b_mu": 0.196,  "T1_ms": 337,  "T2_ms": 98},
    "muscle": {"thickness_cm": 3.0, "density": 0.9,
               "α_db_cm_mhz": 0.54, "c_m_per_s": 1580,
               "m_mu": -0.0065, "b_mu": 0.26,   "T1_ms": 1233, "T2_ms": 37},
    "tumor":  {"thickness_cm": 1.0, "density": 0.8,
               "α_db_cm_mhz": 0.76, "c_m_per_s": 1564,
               "m_mu": -0.0008, "b_mu": 0.25,   "T1_ms": 1100, "T2_ms": 50},
    "nerve":  {"thickness_cm": 0.5, "density": 0.9,
               "α_db_cm_mhz": 0.9,  "c_m_per_s": 1630,
               "m_mu": -0.0065, "b_mu": 0.24,   "T1_ms": 1083, "T2_ms": 78},
}

modalities = {
    "MRI":        {"voxel_volume_mm3": 1.0, "SNR_ref": 20.0},
    "Ultrasound": {"input_intensity_W_cm2": 0.1, "σ_e_W_cm2": 0.001},
    "Xray_CT":    {"tube_voltage_kVp_max": 120},
}

# -------------------------------------------------
# 1. MRI: global optimum CNR on a TR–TE grid (unchanged)
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

σ_ref     = S_muscle.max() / modalities["MRI"]["SNR_ref"]
CNR_grid  = np.abs(S_tumor - S_muscle) / σ_ref
imax      = np.unravel_index(np.argmax(CNR_grid), CNR_grid.shape)
TR_opt, TE_opt, CNR_opt = TR_vals_ms[imax[0]], TE_vals_ms[imax[1]], CNR_grid[imax]

# -------------------------------------------------
# 2. Ultrasound: depth-wise intensity with speckle (unchanged)
# -------------------------------------------------
freq_MHz  = 5.0
depths_cm = np.linspace(0,
                        tissues["fat"]["thickness_cm"]
                        + tissues["muscle"]["thickness_cm"]
                        + tissues["tumor"]["thickness_cm"], 300)

def cumulative_intensity(depth_cm):
    I = 1.0
    remain = depth_cm
    for layer in ["fat", "muscle", "tumor"]:
        if remain <= 0:
            break
        step = min(tissues[layer]["thickness_cm"], remain)
        μ_db = tissues[layer]["α_db_cm_mhz"] * freq_MHz
        I *= 10 ** (-μ_db * step / 10)
        remain -= step
    return I

I_depth        = np.array([cumulative_intensity(d) for d in depths_cm])
σ_e            = modalities["Ultrasound"]["σ_e_W_cm2"]
cv_rayleigh    = np.sqrt((4 - np.pi) / np.pi)          # ≈0.522
σ_speckle      = cv_rayleigh * I_depth
σ_total        = np.sqrt(σ_e**2 + σ_speckle**2)
idx_boundary   = np.searchsorted(depths_cm,
                                 tissues["fat"]["thickness_cm"]
                                 + tissues["muscle"]["thickness_cm"])
I_muscle_layer = I_depth[idx_boundary - 1]
I_tumor_surf   = I_depth[idx_boundary]
σ_muscle       = σ_total[idx_boundary - 1]
CNR_ultrasound = np.abs(I_tumor_surf - I_muscle_layer) / σ_muscle

# -------------------------------------------------
# 3 a. X-ray: polychromatic spectrum & bow-tie filter
# -------------------------------------------------
kVp          = modalities["Xray_CT"]["tube_voltage_kVp_max"]
energies_keV = np.linspace(20, kVp, 400)
dE           = energies_keV[1] - energies_keV[0]

def tungsten_brems_spectrum(E, kVp):
    return np.clip(kVp - E, 0, None) * E          # ∝ E·(kVp−E)

def μ_Al(E):
    return 5e-3 * (30 / E)**3                     # crude Al μ(E)

S_unfilt = tungsten_brems_spectrum(energies_keV, kVp)
S_filt   = S_unfilt * np.exp(-μ_Al(energies_keV) * 1.0)   # 1 cm Al

def μ_tissue(t, E_keV):
    return tissues[t]["m_mu"] * E_keV + tissues[t]["b_mu"]

def I_poly(tissue):
    t_len = tissues[tissue]["thickness_cm"]
    return np.sum(S_filt * np.exp(-μ_tissue(tissue, energies_keV) * t_len)) * dE

I_poly_dict = {t: I_poly(t) for t in tissues}
I_muscle_poly = I_poly_dict["muscle"]
I_tumor_poly  = I_poly_dict["tumor"]
CNR_xray_poly = np.abs(I_tumor_poly - I_muscle_poly) / np.sqrt(I_muscle_poly)

# -------------------------------------------------
# 3 b. X-ray: energy-dependent CNR for *all* pairs
# -------------------------------------------------
def mono_intensity(tissue):
    """Return I(E) (vector) for a single-layer path through <tissue>."""
    t_len = tissues[tissue]["thickness_cm"]
    return np.exp(-μ_tissue(tissue, energies_keV) * t_len)

mono_I = {t: mono_intensity(t) for t in tissues}

pairs_to_plot = [("tumor", "muscle"),
                 ("tumor", "fat"),
                 ("muscle", "fat"),
                 ("nerve", "muscle")]

def cnr_curve(t1, t2):
    I1, I2 = mono_I[t1], mono_I[t2]
    return np.abs(I1 - I2) / np.sqrt(I2)          # Poisson σ ≈ √I₂

cnr_curves = {p: cnr_curve(*p) for p in pairs_to_plot}

# =================================================
# 🪄  TAB-ORGANIZED WINDOW
# =================================================
window = abracatabra.TabbedPlotWindow(window_id="Multi-modal-CNR", ncols=2)

# ---- (1) MRI CNR heat-map ------------------------
fig = window.add_figure_tab("MRI CNR map", col=0)
ax  = fig.add_subplot()
im  = ax.imshow(CNR_grid,
                extent=[TE_vals_ms.min(), TE_vals_ms.max(),
                        TR_vals_ms.max(), TR_vals_ms.min()],
                aspect='auto')
ax.scatter(TE_opt, TR_opt, marker='x', s=80, color='white')
ax.set(xlabel="TE (ms)", ylabel="TR (ms)",
       title=f"MRI CNR vs TR & TE\nopt ≈ {CNR_opt:.1f} @ TR {TR_opt:.0f} ms / TE {TE_opt:.0f} ms")
fig.colorbar(im, ax=ax, label="CNR (tumour – muscle)")

# ---- (2) Ultrasound depth profile ---------------
fig = window.add_figure_tab("Ultrasound depth", col=1)
ax  = fig.add_subplot()
ax.plot(depths_cm, I_depth, label="Mean intensity")
ax.fill_between(depths_cm, I_depth - σ_speckle, I_depth + σ_speckle,
                alpha=0.3, label="±σ (speckle)")
ax.set(xlabel="Depth (cm)", ylabel="Intensity (norm.)",
       title=f"Ultrasound 5 MHz\nCNR(tumour/muscle) ≈ {CNR_ultrasound:.2f}")
ax.legend(); ax.grid(True)

# ---- (3) X-ray spectra --------------------------
fig = window.add_figure_tab("X-ray spectrum")
ax  = fig.add_subplot()
ax.plot(energies_keV, S_unfilt / S_unfilt.max(), label="Unfiltered")
ax.plot(energies_keV, S_filt   / S_filt.max(),   label="After 1 cm Al")
ax.set(xlabel="Energy (keV)", ylabel="Relative intensity",
       title=f"120 kVp, bow-tie filtered\nCNR(tumour/muscle, poly) ≈ {CNR_xray_poly:.2f}")
ax.legend(); ax.grid(True)

# ---- (4) NEW – X-ray CNR curves -----------------
fig = window.add_figure_tab("X-ray CNR curves")
ax  = fig.add_subplot()
for (t1, t2), curve in cnr_curves.items():
    ax.plot(energies_keV, curve, label=f"{t1.capitalize()} vs {t2}")
ax.set(xlabel="Energy (keV)", ylabel="CNR (per keV)",
       title="Energy-dependent CNR for key tissue pairs")
ax.legend(); ax.grid(True)

# Final polish & launch
window.apply_tight_layout()
abracatabra.abracatabra()             # ⭐ replaces plt.show()

# -------------------------------------------------
# Console summary
# -------------------------------------------------
print(f"\n▶ MRI optimum: TR ≈ {TR_opt:.0f} ms, TE ≈ {TE_opt:.0f} ms, CNR ≈ {CNR_opt:.1f}")
print(f"▶ Ultrasound tumour/muscle CNR (with speckle): {CNR_ultrasound:.2f}")
print(f"▶ X-ray tumour/muscle CNR (poly, filtered):    {CNR_xray_poly:.2f}")
for (t1, t2), curve in cnr_curves.items():
    print(f"   – Peak mono-energetic CNR for {t1}/{t2}: {curve.max():.1f} at "
          f"{energies_keV[curve.argmax()]:.0f} keV")
