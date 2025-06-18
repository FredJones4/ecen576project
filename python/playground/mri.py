'''
MRI contrast demo using AbracaTABra tabbed windows
Requires:
    pip install "abracatabra[qt-pyside6]"
'''
import json
import numpy as np
import abracatabra

# -------------------------------------------------
# 1. Load tissue properties & MRI reference
# -------------------------------------------------
# Update these paths to point at your JSON files
tissues_path = r'C:\Users\Owner\code_2025_spring\ecen576\project\python\tissues.json'
modalities_path = r'C:\Users\Owner\code_2025_spring\ecen576\project\python\modalities.json'

with open(tissues_path, 'r') as f:
    tissues = json.load(f)
with open(modalities_path, 'r') as f:
    modalities = json.load(f)

# -------------------------------------------------
# 2. MRI Signal and CNR calculations
# -------------------------------------------------
TR_ms        = 500  # repetition time in ms
TE_values_ms = np.linspace(5, 100, 40)  # echo times for plotting

# Spin-echo signal model
def mri_signal(T1, T2, TR, TE, S0=1.0):
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

# Compute normalized signals for tissues
signals = {t: mri_signal(tissues[t]['T1_ms'], tissues[t]['T2_ms'], TR_ms, TE_values_ms)
           for t in ['fat', 'muscle', 'tumor', 'nerve']}

# Contrast-to-noise ratio helper
def cnr(I_obj, I_bg, sigma):
    return np.abs(I_obj - I_bg) / sigma

# Estimate noise sigma from muscle SNR reference at TE→0
gamma_ref = modalities['MRI']['SNR_ref']
sigma_noise = signals['muscle'][0] / gamma_ref
print(sigma_noise)

# CNR: tumor vs muscle
cnr_tumor_muscle = cnr(signals['tumor'], signals['muscle'], sigma_noise)

# -------------------------------------------------
# 3. MRI Contrast (relative difference)
# -------------------------------------------------
# Contrast vs muscle: (I_tissue - I_muscle) / I_muscle
contrast_vs_muscle = {
    t: (signals[t] - signals['muscle']) / signals['muscle']
    for t in ['fat', 'tumor', 'nerve']
}

# Tumor-to-nerve contrast: (I_tumor - I_nerve) / I_nerve
contrast_tumor_nerve = (signals['tumor'] - signals['nerve']) / signals['nerve']

# =================================================
#           🪄  BUILD THE TABBED PLOT WINDOW
# =================================================
window = abracatabra.TabbedPlotWindow(window_id='MRI-Contrast-Demo', ncols=2)

# ---- MRI Signal vs TE -------------------------------------------------------
fig = window.add_figure_tab('MRI Signal vs TE', col=0)
ax = fig.add_subplot()
for tissue, sig in signals.items():
    ax.plot(TE_values_ms, sig, label=tissue.capitalize())
ax.set(xlabel='Echo time TE (ms)', ylabel='Relative MRI signal',
       title='Spin-Echo MRI Signal (TR = 500 ms)')
ax.grid(True)
ax.legend()

# ---- MRI CNR: Tumor vs Muscle -----------------------------------------------
fig = window.add_figure_tab('MRI CNR Tumor/Muscle', col=1)
ax = fig.add_subplot()
ax.plot(TE_values_ms, cnr_tumor_muscle)
ax.set(xlabel='Echo time TE (ms)', ylabel='CNR',
       title='MRI CNR (Tumor vs Muscle)')
ax.grid(True)

# ---- MRI Contrast vs Muscle -------------------------------------------------
fig = window.add_figure_tab('MRI Contrast Tissue/Muscle', col=0)
ax = fig.add_subplot()
for tissue, contrast in contrast_vs_muscle.items():
    ax.plot(TE_values_ms, contrast, label=tissue.capitalize())
ax.set(xlabel='Echo time TE (ms)', ylabel='Contrast',
       title='MRI Contrast (Tissue vs Muscle)')
ax.grid(True)
ax.legend()

# ---- MRI Contrast: Tumor vs Nerve -------------------------------------------
fig = window.add_figure_tab('MRI Contrast Tumor/Nerve', col=1)
ax = fig.add_subplot()
ax.plot(TE_values_ms, contrast_tumor_nerve)
ax.set(xlabel='Echo time TE (ms)', ylabel='Contrast',
       title='MRI Contrast (Tumor vs Nerve)')
ax.grid(True)

# Finalize layout and show
window.apply_tight_layout()
abracatabra.abracatabra()
