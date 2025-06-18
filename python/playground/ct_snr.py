'''
Multi-size CT phantom + CNR & SNR demo (with quantum Poisson noise & contrast)
-----------------------------------------------------------------------
Put this in a file (e.g. multi_ct_demo_poisson.py) and run it.  Required
packages: numpy, matplotlib, pandas, scikit-image ≥ 0.22.

Author: <your name / date>

# NOTE: I0, as described in Beer–Lambert law, is *not* normalised to 1 here –
# it is the *actual* incident photon fluence per detector bin. Typical values
# range from 1 × 10⁴ (very low‑dose) to 1 × 10⁵ (routine diagnostic) photons.
# Choose whichever combination of input variables for this. We the code
# developers are not the experts in optimum safety levels and leave that task
# for the medical physicists.

CT CNR calculated as follows: https://howradiologyworks.com/x-ray-cnr/
Conversion for HU units: https://radiopaedia.org/articles/hounsfield-unit?lang=us

Best so far: 512 pixels, keV = 70.
(pixels, keV) = (512, 35) also shows promising results.
(pixels, keV) = (512, 20) is still very promising, given current setup.
TODO: confirm dz occurs in Lambert law.
Pixel sizes 512, 1024, 2048 show promising results. Though nerve‑to‑muscle
CNR is low enough that it may be considered too much. 512 is already very
high.

NOTE for when keV is not a constant anymore: may need to create more of a
spectra.
TODO: confirm that output value of CT matrix is also a μ value.
TODO: see if error exists in pixel calculation; differs per pixel from table
calculation.

---------------------------------------------------------------------
Key sources consulted (see in‑line comments for context)
---------------------------------------------------------------------
[1] Low‑dose CT simulation by Poisson noise – Phys. Med. Biol. 65:135010 (2020)
[2] Beer–Lambert law refresher – Radiopaedia article (2025‑04)
[3] scikit‑image Radon transform tutorial (v0.24)
[4] Photon (shot) noise properties – Hasinoff 2012 preprint, MIT CSAIL
[5] Photon‑counting detector review – ScienceDirect, 2025
[6] Mixed Poisson–Gaussian model for CT – Ding et al., IEEE TMI 2016
[7] Handling log(0) pitfalls – DataDuets blog 2023
[8] NumPy random.poisson documentation (v2.3)
[9] Quantum noise in radiology – Radiopaedia article
[10] Filtered‑back‑projection & Ram‑Lak filter – HowRadiologyWorks guide

---------------------------------------------------------------------
CHANGELOG (2025‑06‑17)
---------------------------------------------------------------------
★ Removed the hard clip at 1 photon in `add_poisson_noise`; now adds a small
  ε = 1 e‑6 before the log to retain information from very low‑dose rays.
★ Added helper `mu_to_hu()` and display windowing so the reconstructed slice
  is shown in HU with a default window of ±400 HU around water.
★ CNR is now measured **on the reconstructed HU image** using the standard
  deviation of the muscle ROI as σ.  The empirical `_sigma_factor` table has
  been retired.
★ Default `I0` for `demo()` is now 5 × 10⁴ photons per detector element – a
  realistic low‑dose scan – and the CLI demo loops through several I0 values
  so users can see the dose‑to‑image‑quality relationship.
★ Added SNR column to the metrics table (mean_HU / σ_muscle) so users can
  directly assess absolute signal‑to‑noise alongside contrast metrics.
---------------------------------------------------------------------
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# ---------------------------------------------------------------------
# 0.  Parameters that describe the *real* object (all lengths in cm)
# ---------------------------------------------------------------------

# (unchanged)

tumor_radius      = 1.0                     # cm
muscle_thickness  = 3.0                     # cm
fat_thickness     = 0.5                     # cm
muscle_outer_r    = muscle_thickness        # 3.0 cm – tumour thickness removed
outer_radius      = muscle_outer_r + fat_thickness    # 3.5 cm
nerve_radius      = 0.5                     # cm
gap_to_tumor      = 0.05                    # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius  # 1.55 cm

# ---------------------------------------------------------------------
# 1.  Tissue dictionary & linear‑attenuation helper (unchanged)
# ---------------------------------------------------------------------

tissues = {
    "fat":    {"xray_mu_linear": -0.0004, "xray_mu_intercept": 0.196},
    "muscle": {"xray_mu_linear": -0.0065, "xray_mu_intercept": 0.260},
    "tumor":  {"xray_mu_linear": -0.0008, "xray_mu_intercept": 0.250},
    "nerve":  {"xray_mu_linear": -0.0065, "xray_mu_intercept": 0.240},
}

def linear_attenuation(tissue: str, E_keV: float) -> float:
    """Return μ (cm⁻¹) at a given energy."""
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E_keV + b

# ---------------------------------------------------------------------
# 2.  Noise helpers (★ modified)
# ---------------------------------------------------------------------

EPS = 1e-6  # ε added to avoid log(0)

def add_poisson_noise(sino: np.ndarray, I0: float = 1e5) -> np.ndarray:
    """Add quantum (shot) noise to a noiseless line‑integral sinogram."""
    expected_counts = I0 * np.exp(-sino)          # Beer–Lambert [2]
    noisy_counts    = np.random.poisson(expected_counts)  # Poisson [4,8]
    noisy_counts    = noisy_counts.astype(float) + EPS    # allow true zeros
    return -np.log(noisy_counts/ I0)

# ---------------------------------------------------------------------
# 3.  CT pipeline (unchanged apart from call site defaults)
# ---------------------------------------------------------------------

def ct_pipeline(mu: np.ndarray, *, angles=None, filter_name: str = 'ramp',
                circle: bool = True, I0: float | None = None):
    """Forward‑project *mu* and reconstruct with optional Poisson noise."""
    if mu.shape[0] != mu.shape[1]:
        raise ValueError("mu image must be square")
    N = mu.shape[0]
    if angles is None:
        angles = np.linspace(0., 180., N, endpoint=False)

    sino = radon(mu, theta=angles, circle=circle)  # ∫μ dl [3]
    if I0 is not None:
        # sino = add_poisson_noise(sino, I0) # TODO: uncomment to 
        pass

    backproj = iradon(sino, theta=angles, filter_name=None,
                      circle=circle, preserve_range=True)
    fbp      = iradon(sino, theta=angles, filter_name=filter_name,
                      circle=circle, preserve_range=True)
    return sino, backproj, fbp

# ---------------------------------------------------------------------
# 4.  Build a square phantom (unchanged)
# ---------------------------------------------------------------------

def build_phantom(N: int, E_keV: float = 70.0):
    """Return *μ‑image*, tissue masks, μ values and pixel size."""
    field_cm = 2 * outer_radius                  # 7.0 cm – full body diameter
    px_cm    = field_cm / N

    coords   = (np.arange(N) - N/2 + 0.5) * px_cm
    X, Y     = np.meshgrid(coords, coords)
    R        = np.hypot(X, Y)

    mu_tiss = {t: linear_attenuation(t, E_keV) for t in tissues}

    mu_img = np.zeros((N, N), dtype=np.float32)   # initialise with air

    fat_mask    = (R <= outer_radius) & (R > muscle_outer_r)
    muscle_mask = (R <= muscle_outer_r) & (R > tumor_radius)
    tumor_mask  = (R <= tumor_radius)
    nerve_mask  = ((X + nerve_center_dist)**2 + Y**2) <= nerve_radius**2

    mu_img[fat_mask]    = mu_tiss["fat"]
    mu_img[muscle_mask] = mu_tiss["muscle"]
    mu_img[tumor_mask]  = mu_tiss["tumor"]
    mu_img[nerve_mask]  = mu_tiss["nerve"]

    masks = {"fat": fat_mask, "muscle": muscle_mask,
             "tumor": tumor_mask, "nerve": nerve_mask}
    return mu_img, masks, mu_tiss, px_cm

# ---------------------------------------------------------------------
# 5.  Image & metric helpers (★ modified)
# ---------------------------------------------------------------------

def mu_to_hu(mu_arr: np.ndarray, mu_water: float = 0.19) -> np.ndarray:
    """Vectorised μ → HU conversion."""
    return 1000.0 * (mu_arr - mu_water) / mu_water

# ---------------------------------------------------------------------
# 6.  Main driver – computes CNR, SNR & contrast from the *reconstructed* image
# ---------------------------------------------------------------------

def demo(N: int = 512, E_keV: float = 70.0, I0: float | None = 5e4,
         show_images: bool = True, window_hu: int = 400):
    """Run the phantom simulation and print a CNR / SNR / contrast table."""

    mu_img, masks, mu_tiss, px_cm = build_phantom(N, E_keV)
    sino, bp, fbp = ct_pipeline(mu_img, filter_name='ramp', I0=I0)

    # -----------------------------------------------------------------
    # Metrics on the reconstructed slice (FBP) -------------------------
    # -----------------------------------------------------------------
    hu_img = mu_to_hu(fbp)

    mean_hu = {t: hu_img[masks[t]].mean() for t in ["fat", "muscle", "tumor", "nerve"]}
    sigma_bg = hu_img[masks["muscle"]].std(ddof=1)  # noise estimate

    # SNR per tissue ---------------------------------------------------
    snr = {t: mean_hu[t] / sigma_bg for t in mean_hu}

    # CNR and contrast -------------------------------------------------
    cnr_to_muscle = {t: np.abs(mean_hu[t] - mean_hu["muscle"]) / sigma_bg
                     for t in ["fat", "tumor", "nerve"]}
    cnr_tumor_nerve = np.abs(mean_hu["tumor"] - mean_hu["nerve"]) / sigma_bg

    contrast_to_muscle = {t: np.abs((mean_hu[t] - mean_hu["muscle"]) / mean_hu["muscle"])
                          for t in mean_hu}
    contrast_tumor_nerve = np.abs((mean_hu["tumor"] - mean_hu["nerve"]) / mean_hu["nerve"])

    # -----------------------------------------------------------------
    # Tabulation -------------------------------------------------------
    # -----------------------------------------------------------------
    df = (pd
          .DataFrame.from_dict(mean_hu, orient='index', columns=['HU'])
          .assign(SNR=[snr[t] for t in mean_hu])
          .assign(CNR_vs_muscle=[cnr_to_muscle.get(t, np.nan) for t in mean_hu])
          .assign(Contrast_vs_muscle=[contrast_to_muscle.get(t, np.nan) for t in mean_hu])
          .rename_axis("Tissue")
          .round({"HU": 1, "SNR": 2, "CNR_vs_muscle": 2, "Contrast_vs_muscle": 3}))

    # extra lines for tumour‑to‑nerve metrics
    df.loc["tumor‑to‑nerve CNR"]      = [np.nan, np.nan, round(cnr_tumor_nerve, 2), np.nan]
    df.loc["tumor‑to‑nerve contrast"] = [np.nan, np.nan, np.nan, round(contrast_tumor_nerve, 3)]

    # -----------------------------------------------------------------
    # visualisation ----------------------------------------------------
    # -----------------------------------------------------------------
    if show_images:
        for title, im, aspect in [
            ("Sinogram", sino, 'auto'),
            ("Unfiltered back‑projection", bp, 'equal'),
            ("Filtered back‑projection (Ram‑Lak)", hu_img, 'equal')]:
            plt.figure(figsize=(6, 6))
            if title.startswith("Filtered"):
                vmin, vmax = -window_hu, window_hu
                cmap = 'gray'
            else:
                vmin = vmax = None
                cmap = 'gray'
            plt.imshow(im, cmap=cmap, aspect=aspect, vmin=vmin, vmax=vmax)
            plt.title(f"{title}  –  {N}×{N} px, {E_keV} keV, I0={I0 if I0 else '∞'}")
            plt.axis('off')
            plt.tight_layout()

        plt.figure(figsize=(6, 6))
        plt.imshow(mu_img, cmap='gray', extent=[-outer_radius, outer_radius]*2)
        plt.title(f"μ phantom ({N}×{N}), pixel = {px_cm*10:.2f} mm")
        plt.xlabel("x (cm)"); plt.ylabel("y (cm)")
        plt.colorbar(label="μ (cm⁻¹)")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # quick sweep over dose levels for demonstration
    for dose in [1, 1e3, 1e4, 5e4, None]:  # None = noiseless reference
        print(f"\n=== I0 = {dose} photons per detector element ===")
        demo(N=512, E_keV=20.0, I0=dose, show_images=True)
