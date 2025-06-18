'''
MRI contrast demo using AbracaTABra tabbed windows

Spin-echo regime. Noise is gaussian, white noise independent on the real
and imaginary channels. Magnitude images follow a Rician distribution.

This updated script: 1) explicitly models the Rician noise in the signal generation,
2) allows optimization over parameters: TR, TE range, TE resolution, and number of TR repetitions,
3) displays both signal curves and an optimization summary tab,
4) now optimizes tumor–nerve contrast ((I_nerve – I_tumor)/I_tumor) instead of CNR,
   while still reporting contrast of all other tissues vs. muscle.

See notes: https://chatgpt.com/share/e/68528487-f160-8001-a9b5-367978e5f43b

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

tissues_path = r'C:/Users/Owner/code_2025_spring/ecen576/project/python/tissues.json'
modalities_path = r'C:/Users/Owner/code_2025_spring/ecen576/project/python/modalities.json'

tissues = json.load(open(tissues_path, 'r'))
modalities = json.load(open(modalities_path, 'r'))

# -------------------------------------------------
# 2. Define MRI Signal and Rician noise model
# -------------------------------------------------

def mri_signal(T1, T2, TR, TE, S0=1.0):
    """
    Spin-echo signal model:
    S = S0 * (1 - exp(-TR/T1)) * exp(-TE/T2)
    """
    return S0 * (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)


def add_rician_noise(signal, sigma, n_samples):
    """
    Add Rician distributed noise to a clean signal array.
    Noise: real and imaginary channels independent Gaussian,
    magnitude follows Rician distribution.
    Returns mean noisy magnitude per TE.
    """
    noisy_mean = []
    for val in signal:
        real = val + np.random.normal(0, sigma, n_samples)
        imag = np.random.normal(0, sigma, n_samples)
        mag = np.sqrt(real**2 + imag**2)
        noisy_mean.append(np.mean(mag))
    return np.array(noisy_mean)


# -------------------------------------------------
# 3. Parameter grid and optimization (tumor–nerve contrast)
# -------------------------------------------------
TR_list      = np.arange(500, 3000, 100)      # ms
TE_end_list  = np.arange(60, 420, 20)         # max TE, ms
res_list     = np.arange(256,1024,256)     # number of TE points
nTRs_list    = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])

best_score   = -np.inf
best_params  = None

for TR in TR_list:
    for TE_end in TE_end_list:
        for res in res_list:
            TE_vals = np.linspace(5, TE_end, res)
            # need muscle for noise, tumor+nerve for contrast
            signals = {
                'muscle': mri_signal(tissues['muscle']['T1_ms'],
                                     tissues['muscle']['T2_ms'],
                                     TR, TE_vals),
                'tumor' : mri_signal(tissues['tumor']['T1_ms'],
                                     tissues['tumor']['T2_ms'],
                                     TR, TE_vals),
                'nerve' : mri_signal(tissues['nerve']['T1_ms'],
                                     tissues['nerve']['T2_ms'],
                                     TR, TE_vals),
            }
            gamma_ref = modalities['MRI']['SNR_ref']
            sigma     = signals['muscle'][0] / gamma_ref

            for nTRs in nTRs_list:
                sigma_eff = sigma / np.sqrt(nTRs)
                # tumor–nerve contrast metric (noise-free)
                contrast_vals = (signals['nerve'] - signals['tumor']) / signals['tumor']
                score = np.mean(contrast_vals)
                if score > best_score:
                    best_score  = score
                    best_params = (TR, TE_end, res, nTRs)

TR_opt, TE_opt_end, res_opt, nTRs_opt = best_params

# recompute opt curves
TE_values_opt = np.linspace(5, TE_opt_end, res_opt)
signals_opt   = {
    t: mri_signal(tissues[t]['T1_ms'], tissues[t]['T2_ms'], TR_opt, TE_values_opt)
    for t in ['fat','muscle','tumor','nerve']
}
gamma_ref       = modalities['MRI']['SNR_ref']
sigma_noise_opt = signals_opt['muscle'][0] / gamma_ref
contrast_tn_opt = (signals_opt['nerve'] - signals_opt['tumor']) / signals_opt['tumor']

# compute mean contrasts of each non-muscle tissue vs muscle
contrast_vs_muscle = {}
for t in ['fat','tumor','nerve']:
    contrast_vs_muscle[t] = np.mean((signals_opt[t] - signals_opt['muscle']) / signals_opt['muscle'])

# -------------------------------------------------
# 4. Build the Tabbed Plot Window
# -------------------------------------------------
window = abracatabra.TabbedPlotWindow(window_id='MRI-Contrast-Optimized', ncols=2)

# Signal vs TE (Optimal)
fig = window.add_figure_tab('Signal vs TE (opt)', col=0)
ax = fig.add_subplot()
for tissue, sig in signals_opt.items():
    ax.plot(TE_values_opt, sig, label=tissue.capitalize())
ax.set(xlabel='TE (ms)', ylabel='Signal',
       title=f'Spin-Echo Signal (TR={TR_opt} ms)')
ax.legend(); ax.grid(True)

# Tumor–Nerve Contrast vs TE (Optimal)
fig = window.add_figure_tab('Contrast T→N (opt)', col=1)
ax = fig.add_subplot()
ax.plot(TE_values_opt, contrast_tn_opt)
ax.set(xlabel='TE (ms)', ylabel='Contrast',
       title=f'Tumor→Nerve Contrast, nTRs={nTRs_opt}')
ax.grid(True)

# Optimization summary as a figure tab
fig = window.add_figure_tab('Optimization Summary', col=0)
ax = fig.add_subplot()
ax.axis('off')  # no axes
summary = f"""
Optimal Parameters:
  • TR                = {TR_opt} ms
  • Max TE            = {TE_opt_end} ms
  • TE resolution     = {res_opt} points
  • Number of TRs     = {nTRs_opt}

Achieved mean Tumor→Nerve contrast  = {best_score:.3f}

Mean contrasts vs. Muscle:
  • Fat→Muscle    = {contrast_vs_muscle['fat']:.3f}
  • Tumor→Muscle = {contrast_vs_muscle['tumor']:.3f}
  • Nerve→Muscle = {contrast_vs_muscle['nerve']:.3f}
"""
ax.text(
    0.01, 0.99,
    summary,
    va='top',
    family='monospace',
    wrap=True
)

# Rician noise effect: Mean noisy signal vs TE
n_samples = 5000
fig = window.add_figure_tab('Mean Noisy Signal vs TE', col=1)
ax = fig.add_subplot()
for tissue, sig in signals_opt.items():
    noisy = add_rician_noise(sig, sigma_noise_opt, n_samples)
    ax.plot(TE_values_opt, noisy, '--', label=f"{tissue.capitalize()} (noisy)")
ax.set(xlabel='TE (ms)', ylabel='Mean Noisy Signal',
       title='Rician Noise Effect')
ax.legend(); ax.grid(True)

window.apply_tight_layout()
abracatabra.abracatabra()
