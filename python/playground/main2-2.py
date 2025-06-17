"""
Medical‑imaging contrast demo using AbracaTABra tabbed windows
Requires:
    pip install "abracatabra[qt‑pyside6]"   # or any other supported Qt binding

This demo visualises *contrast* (plain signal/intensity differences) **separately** from *contrast‑to‑noise ratio (CNR)* for MRI, X‑ray/CT and Ultrasound.
"""
# TODO: leave all comments in place; new comments explain additional tabs
# TODO: MRI CNR compares Tumor ↔ Nerve (noise from muscle @ TE→0)
# TODO: X‑ray/CT and Ultrasound now plot CNR and contrast on distinct tabs

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import abracatabra                      # ⭐ NEW: tabbed Qt windows

# -------------------------------------------------
# 1. Load tissue properties & modality parameters
# -------------------------------------------------
file_path_tissues     = r"C:\Users\Owner\code_2025_spring\ecen576\project\python\tissues.json"
file_path_modalities  = r"C:\Users\Owner\code_2025_spring\ecen576\project\python\modalities.json"

with open(file_path_tissues, 'r') as f:
    tissues = json.load(f)
with open(file_path_modalities, 'r') as f:
    modalities = json.load(f)

# -------------------------------------------------
# 2. Generic helper functions
# -------------------------------------------------

def contrast(I_object, I_background):
    """Plain (noise‑agnostic) contrast = |I_obj − I_back|"""
    return np.abs(I_object - I_background)

def cnr(I_object, I_background, sigma):
    """Contrast‑to‑Noise Ratio"""
    return np.abs(I_object - I_background) / sigma

# -------------------------------------------------
# 3. MRI – Spin‑Echo signal, contrast & CNR vs TE
# -------------------------------------------------
TR_ms        = 500                                    # repetition time (ms)
TE_values_ms = np.linspace(5, 100, 40)                # echo times for plotting

# Normalised spin‑echo signal (see Radiopaedia & RIT CIS MRI text)

def mri_signal(T1, T2, TR, TE, S0=1.0):
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

signals = {t: mri_signal(tissues[t]["T1_ms"], tissues[t]["T2_ms"],
                         TR_ms, TE_values_ms)
           for t in ["fat", "muscle", "tumor", "nerve"]}

# MRI contrast (Tumor vs Nerve)
contrast_mri = contrast(signals["tumor"], signals["nerve"])

# Noise σ derived from muscle SNR_ref at TE→0
sigma_noise_mri = signals["muscle"][0] / modalities["MRI"]["SNR_ref"]
CNR_mri         = cnr(signals["tumor"], signals["nerve"], sigma_noise_mri)

# -------------------------------------------------
# 4. X‑ray / CT – transmission, contrast & CNR vs energy
# -------------------------------------------------
energies_keV = np.linspace(20, 150, 200)

# Linear attenuation μ(E) ≈ m·E + b (see FA Duck, 1990)

def mu(tissue, E_keV):
    return tissues[tissue]["xray_mu_linear"] * E_keV + tissues[tissue]["xray_mu_intercept"]

# Tube settings (max) – note: simplistic scaling for demo purposes
modalities["Xray_CT"]["tube_current_A_max"] = 2
I0 = (modalities["Xray_CT"]["tube_voltage_kVp_max"]
      * modalities["Xray_CT"]["tube_current_A_max"]
      * modalities["Xray_CT"]["conversion_efficiency"]
      * modalities["Xray_CT"]["detector_efficiency"])

# Intensities through different paths
I_muscle = I0 * np.exp(-mu("muscle", energies_keV) * tissues["muscle"]["thickness_cm"])
I_tumor  = I0 * np.exp(-(mu("tumor", energies_keV) * tissues["tumor"]["thickness_cm"] +
                         mu("muscle", energies_keV) * (tissues["muscle"]["thickness_cm"] - tissues["tumor"]["thickness_cm"])) )
I_nerve  = I0 * np.exp(-(mu("nerve", energies_keV) * tissues["nerve"]["thickness_cm"] +
                         mu("muscle", energies_keV) * (tissues["muscle"]["thickness_cm"] - tissues["nerve"]["thickness_cm"])) )

# X‑ray/CT contrast & CNR (Tumor vs Muscle) minus (Nerve vs Muscle)
contrast_xray = contrast(I_tumor, I_muscle)
CNR_xray      = (cnr(I_tumor, I_muscle, np.sqrt(I_muscle))
                 - cnr(I_nerve, I_muscle, np.sqrt(I_muscle)))

# -------------------------------------------------
# 5. Ultrasound – intensity, contrast & CNR vs depth
# -------------------------------------------------
freq_MHz  = 5.0
max_depth = (tissues["fat"]["thickness_cm"] +
             tissues["muscle"]["thickness_cm"] +
             tissues["tumor"]["thickness_cm"])

depths_cm = np.linspace(0, max_depth, 300)


def cumulative_intensity(depth_cm):
    I      = I0  # reuse X‑ray I0 as arbitrary starting acoustic power (demo only)
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

# Contrast & CNR at the fat↔muscle boundary
boundary_idx      = np.searchsorted(depths_cm,
                                    tissues["fat"]["thickness_cm"] + tissues["muscle"]["thickness_cm"])
I_muscle_layer    = I_depth[boundary_idx - 1]
I_tumor_surface   = I_depth[boundary_idx]
contrast_ultrasnd = contrast(I_tumor_surface, I_muscle_layer)
CNR_ultrasnd      = cnr(I_tumor_surface, I_muscle_layer,
                       modalities["Ultrasound"]["electronic_noise_std_W_cm2"])

# -------------------------------------------------
# 6. 🪄  Build the tabbed plot window (contrast & CNR split)
# -------------------------------------------------
window = abracatabra.TabbedPlotWindow(window_id="Imaging‑Contrast‑Demo", ncols=2)

# ---- MRI: contrast (signals) ---------------------------------------------------
fig = window.add_figure_tab("MRI Signals vs TE", col=0)
ax  = fig.add_subplot()
for tissue, sig in signals.items():
    ax.plot(TE_values_ms, sig, label=tissue.capitalize())
ax.set(xlabel="Echo time TE (ms)", ylabel="Relative MRI signal",
       title="MRI Signal (TR = 500 ms)")
ax.grid(True)
ax.legend()

# ---- MRI: CNR ------------------------------------------------------------------
fig = window.add_figure_tab("MRI CNR Tumor↔Nerve", col=1)
ax  = fig.add_subplot()
ax.plot(TE_values_ms, CNR_mri)
ax.set(xlabel="Echo time TE (ms)", ylabel="CNR",
       title="MRI CNR (Tumor vs Nerve)")
ax.grid(True)

# ---- X‑ray/CT: contrast --------------------------------------------------------
fig = window.add_figure_tab("X‑ray Transmission", col=0)
ax  = fig.add_subplot()
ax.plot(energies_keV, I_muscle, label="Muscle") # TODO: update for contrast
ax.plot(energies_keV, I_tumor,  label="Muscle + Tumor")
ax.set(xlabel="Energy (keV)", ylabel="Transmitted intensity (norm.)",
       title="X‑ray/CT Transmission vs Energy")
ax.grid(True)
ax.legend()

# ---- X‑ray/CT: CNR -------------------------------------------------------------
fig = window.add_figure_tab("X‑ray CNR Δ(Tumor,Muscle)‑Δ(Nerve,Muscle)", col=1)
ax  = fig.add_subplot()
ax.plot(energies_keV, CNR_xray)
ax.set(xlabel="Energy (keV)", ylabel="CNR",
       title="X‑ray/CT CNR Difference")
ax.grid(True)

# ---- Ultrasound: contrast (intensity profile) ----------------------------------
fig = window.add_figure_tab("Ultrasound Intensity vs Depth", col=0)
ax  = fig.add_subplot()
ax.plot(depths_cm, I_depth)
ax.set(xlabel="Depth (cm)", ylabel="Intensity (norm.)",
       title="Ultrasound Intensity (5 MHz)")
ax.grid(True)

# ---- Ultrasound: CNR -----------------------------------------------------------
fig = window.add_figure_tab("Ultrasound CNR (Tumor vs Muscle)", col=1)
ax  = fig.add_subplot()
ax.plot([0, max_depth], [CNR_ultrasnd]*2)  # horizontal line
ax.set(xlabel="Depth (cm)", ylabel="CNR",
       title=f"Ultrasound CNR = {CNR_ultrasnd:.2f}")
ax.set_ylim(0, max(CNR_ultrasnd*1.2, 1))
ax.grid(True)

# ---- Final tweaks --------------------------------------------------------------
window.apply_tight_layout()
abracatabra.abracatabra()               # ⭐ NEW – replaces plt.show()
