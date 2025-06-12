"""
Medical-imaging contrast demo using AbracaTABra tabbed windows
Requires:
    pip install "abracatabra[qt-pyside6]"   # or any other supported Qt binding
"""

import numpy as np
import abracatabra                      # ⭐ NEW
# Matplotlib is still needed, but we let AbracaTABra create the figures
import matplotlib.pyplot as plt
# -------------------------------------------------
# 1. Dictionaries of tissue properties & modalities
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
    "MRI":        {"voxel_volume_mm3": 1.0,  "SNR_ref": 20.0, "noise_distribution": "Rician"},
    "Ultrasound": {"input_intensity_W_cm2": 0.1, "electronic_noise_std_W_cm2": 0.001,
                   "noise_distribution": "Gaussian + Rayleigh"},
    "Xray_CT":    {"noise_distribution": "Poisson", "tube_voltage_kVp_max": 120,
                   "tube_current_A_max": 300, "conversion_efficiency": 0.01,
                   "detector_efficiency": 1},
} 
def cnr(I_object, I_background, sigma):
    """Function for readability to generate CNR."""
    return np.abs(I_object - I_background)/sigma


# -------------------------------------------------
# 2. MRI Contrast-to-Noise
# -------------------------------------------------
TR_ms         = 500 # repetition time (ms)
TE_values_ms  = np.linspace(5, 100, 40) # echo times for plotting
  
def mri_signal(T1, T2, TR, TE, S0=1.0): # Normalized S0
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)
# For details and reasons behind the above calculation, see: https://radiopaedia.org/articles/spin-echo-sequences  || and also
# https://www.cis.rit.edu/htbooks/mri/chap-10/chap-10.htm#:~:text=The%20signal%20equations%20for%20the,Inversion%20Recovery%20(180%2D90)


signals = {t: mri_signal(tissues[t]["T1_ms"], tissues[t]["T2_ms"],
                         TR_ms, TE_values_ms)
           for t in ["fat", "muscle", "tumor", "nerve"]}
# Contrast-to-noise ratio (CNR) between tumor and muscle
# Assume noise σ_ref gives SNR_ref=20 at TE→0 for muscle
sigma_noise         = signals["muscle"][0] / modalities["MRI"]["SNR_ref"] # page 77 of textbook, solve for σ_ref based on first value
cnr_tumor_muscle    = cnr(signals["tumor"], signals["muscle"], sigma_noise)
#np.abs(signals["tumor"] - signals["muscle"]) / sigma_noise  # This is an array of values; see also HW 9

# -------------------------------------------------
# 3. X-ray Contrast-to-Noise
# -------------------------------------------------
#TODO: investigate Aluminum filter, as seen in main2_abrac.py, considering updates to CNR
energies_keV           = np.linspace(20, 150, 200) #NOTE: changed max from 120 to 150
thickness_cm_path      = tissues["muscle"]["thickness_cm"] + tissues["tumor"]["thickness_cm"] # unused

def linear_attenuation(tissue, E_keV): # See p5/6 of project, with numbers
    # taken from Physical Properties of Tissue: A Comprehensive Reference Book by FA Duck (1990). Made available to the class and available
    #online by the Harold B Lee Library. 
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E_keV + b


I0          = (modalities["Xray_CT"]["tube_voltage_kVp_max"]*modalities["Xray_CT"]["tube_current_A_max"])\
                *modalities["Xray_CT"]["conversion_efficiency"]*modalities["Xray_CT"]["detector_efficiency"] 
# I0 was 1.0, but ... calculating P = IV -> P_actual=P*conv_eff*detector_eff goes with the assumptions given.
I_muscle    = I0 * np.exp(-linear_attenuation("muscle", energies_keV) * #TODO: update the ∑(μ * x) path based on where we are imaging through
                          # TODO: see if we can make the assumption that CT and X-ray will have the same CNR, given different paths all around
                          tissues["muscle"]["thickness_cm"])
I_tumor     = I0 * np.exp(-(linear_attenuation("tumor",  energies_keV) *
                          tissues["tumor"]["thickness_cm"]  +
                          linear_attenuation("muscle", energies_keV) *
                          tissues["muscle"]["thickness_cm"]))
I_nerve     = I0 * np.exp(-(linear_attenuation("nerve",  energies_keV) *
                          tissues["nerve"]["thickness_cm"]  +
                          linear_attenuation("muscle", energies_keV) *
                          tissues["muscle"]["thickness_cm"]))
snr_muscle  = I_muscle / np.sqrt(I_muscle) # mu(μ)=I. Since we assume noise follows Poisson Distribution, σ=√μ=√I. and SNR= μ/σ = I/√I.
#TODO: See if noise must be caluclated for all body flesh, not just the background (muscle)
#NOTE: the above equation could be simplified, but is kept in this format for readability. In fact, the line of code only exists for readability.

cnr_xray    = np.abs(
    cnr(I_tumor, I_muscle, np.sqrt(I_muscle)) 
    - cnr(I_nerve, I_muscle, np.sqrt(I_muscle))
    )

# CNR_xray = cnr_nerve_to_muscle - cnr_tumor_to_muscle
# https://www.sciencedirect.com/topics/nursing-and-health-professions/contrast-to-noise-ratio#:~:text=%5B30%5D-,CNR,b,-where%20the%20numerator

# -------------------------------------------------
# 4. Ultrasound attenuation profile
# -------------------------------------------------
freq_MHz   = 5.0
depths_cm  = np.linspace(0,
                         tissues["fat"]["thickness_cm"] + tissues["muscle"]["thickness_cm"]
                         + tissues["tumor"]["thickness_cm"],
                         300)

def cumulative_intensity(depth_cm):
    I      = I0
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

I_depth          = np.array([cumulative_intensity(d) for d in depths_cm])
sigma_e          = modalities["Ultrasound"]["electronic_noise_std_W_cm2"]
boundary_index   = np.searchsorted(depths_cm,
                                   tissues["fat"]["thickness_cm"] + tissues["muscle"]["thickness_cm"])
I_muscle_layer   = I_depth[boundary_index - 1]
I_tumor_surface  = I_depth[boundary_index]
cnr_ultrasound   = cnr(I_tumor_surface, I_muscle_layer, sigma_e)#np.abs(I_tumor_surface - I_muscle_layer) / sigma_e

# =================================================
#           🪄  BUILD THE TABBED PLOT WINDOW
# =================================================
window = abracatabra.TabbedPlotWindow(window_id="Imaging-Contrast-Demo",
                                      ncols=2)         # 2 columns of tabs

# ---- MRI signal ----------------------------------------------------------------
fig = window.add_figure_tab("MRI signal vs TE", col=0)
ax  = fig.add_subplot()
for tissue, sig in signals.items():
    ax.plot(TE_values_ms, sig, label=tissue.capitalize())
ax.set(xlabel="Echo time TE (ms)", ylabel="Relative MRI signal",
       title="Spin-Echo MRI Signal (TR = 500 ms)")
ax.grid(True)
ax.legend()

# ---- MRI CNR -------------------------------------------------------------------
fig = window.add_figure_tab("MRI CNR Tumor/Muscle", col=1)
ax  = fig.add_subplot()
ax.plot(TE_values_ms, cnr_tumor_muscle)
ax.set(xlabel="Echo time TE (ms)", ylabel="CNR",
       title="MRI CNR (Tumor – Muscle)")
ax.grid(True)

# ---- X-ray transmission --------------------------------------------------------
fig = window.add_figure_tab("X-ray Transmission", col=0)
ax  = fig.add_subplot()
ax.plot(energies_keV, I_muscle, label="Muscle")
ax.plot(energies_keV, I_tumor,  label="Muscle + Tumor")
ax.set(xlabel="Energy (keV)", ylabel="Transmitted intensity (norm.)",
       title="X-ray Transmission vs Energy")
ax.grid(True)
ax.legend()

# ---- X-ray CNR -----------------------------------------------------------------
fig = window.add_figure_tab("X-ray: CNR Tumor/Muscle - CNR Nerve / Muscle", col=1)
ax  = fig.add_subplot()
ax.plot(energies_keV, cnr_xray)
ax.set(xlabel="Energy (keV)", ylabel="CNR",
       title="X-ray CNR (Tumor/Muscle) - CNR(Nerve/Muscle)")
ax.grid(True)

# ---- Ultrasound depth profile --------------------------------------------------
fig = window.add_figure_tab("Ultrasound Intensity vs Depth")
ax  = fig.add_subplot()
ax.plot(depths_cm, I_depth)
ax.set(xlabel="Depth (cm)", ylabel="Intensity (norm.)",
       title="Ultrasound Intensity (5 MHz)")
ax.grid(True)

# Nicely pack sub-plots inside each tab
window.apply_tight_layout()

# Display the tabbed window (blocks until closed)
abracatabra.abracatabra()               # ⭐ NEW – replaces plt.show()
