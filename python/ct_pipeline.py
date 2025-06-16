"""
Multi-size CT phantom + CNR demo
--------------------------------
Put this in a file (e.g. multi_ct_demo.py) and run it.  Required
packages: numpy, matplotlib, pandas, scikit-image 0.22+.

Author: <your name / date>
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

# ---------------------------------------------------------------------
# 0.  Parameters that describe the *real* object (all lengths in cm)
# ---------------------------------------------------------------------
tumor_radius      = 1.0                     # cm
muscle_thickness  = 3.0                     # cm
fat_thickness     = 0.5                     # cm
muscle_outer_r    = muscle_thickness   # 3.0  cm # tumor-thickness removed, since it is included in the tumor circle
outer_radius      = muscle_outer_r + fat_thickness    # 3.5  cm
nerve_radius      = 0.5                     # cm
gap_to_tumor      = 0.05                    # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius  # 1.55 cm

# ---------------------------------------------------------------------
# 1.  Tissue dictionary & linear-attenuation helper
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
# 2.  CT pipeline exactly as in your reference implementation
# ---------------------------------------------------------------------
def ct_pipeline(mu, *, angles=None, filter_name='ramp', circle=True):
    if mu.shape[0] != mu.shape[1]:
        raise ValueError("mu image must be square")
    N = mu.shape[0]
    if angles is None:
        angles = np.linspace(0., 180., N, endpoint=False)

    sino     = radon(mu, theta=angles, circle=circle)
    backproj = iradon(sino, theta=angles, filter_name=None,
                      circle=circle, preserve_range=True)
    fbp      = iradon(sino, theta=angles, filter_name=filter_name,
                      circle=circle, preserve_range=True)
    return sino, backproj, fbp

# ---------------------------------------------------------------------
# 3.  Build a square phantom at a chosen matrix size
# ---------------------------------------------------------------------
def build_phantom(N: int, E_keV: float = 70.0):
    """
    Returns:
        mu_img  – (N,N) array of μ values
        masks   – dict of boolean masks per tissue
        mu_tiss – dict of single μ numbers per tissue
        px_cm   – physical pixel size in cm
    """
    # physical pixel spacing so that the *full* body fits with a 1-px margin
    field_cm = 2 * outer_radius                  # 9.0 cm
    px_cm    = field_cm / N

    # coordinate grid centred at (0,0)
    coords   = (np.arange(N) - N/2 + 0.5) * px_cm
    X, Y     = np.meshgrid(coords, coords)
    R        = np.hypot(X, Y)

    # helper – μ for every tissue at this energy
    mu_tiss = {t: linear_attenuation(t, E_keV) for t in tissues}

    # initialise with air (μ ≈ 0)
    mu_img = np.zeros((N, N), dtype=np.float32)

    # masks in order of *outer → inner* and nerve last to overwrite
    fat_mask    = (R <= outer_radius) & (R > muscle_outer_r)
    muscle_mask = (R <= muscle_outer_r) & (R > tumor_radius)
    tumor_mask  = (R <= tumor_radius)

    # nerve centre left of tumour
    nerve_mask = ((X + nerve_center_dist)**2 + Y**2) <= nerve_radius**2

    # assign μ values
    mu_img[fat_mask]    = mu_tiss["fat"]
    mu_img[muscle_mask] = mu_tiss["muscle"]
    mu_img[tumor_mask]  = mu_tiss["tumor"]
    mu_img[nerve_mask]  = mu_tiss["nerve"]

    masks = {"fat": fat_mask, "muscle": muscle_mask,
             "tumor": tumor_mask, "nerve": nerve_mask}
    return mu_img, masks, mu_tiss, px_cm

# ---------------------------------------------------------------------
# 4.  Contrast-to-noise ratio helpers
# ---------------------------------------------------------------------
_sigma_factor = {512: 1.0, 1024: 1.5, 2048: 3.0}     # given rule

def hounsfield(mu: float, mu_water: float = 0.19) -> float:
    """Convert μ to HU.  Default μ_water ≈ 0.19 cm⁻¹ at 70 keV."""
    return 1000.0 * (mu - mu_water) / mu_water

def cnr(I_obj, I_bg, N_pix: int):
    """CNR with σ = √I_bg scaled by pixel-dependent factor."""
    sigma = np.sqrt(I_bg) * _sigma_factor[N_pix]
    return np.abs(I_obj - I_bg) / sigma

# ---------------------------------------------------------------------
# 5.  Main driver
# ---------------------------------------------------------------------
def demo(N: int = 512, E_keV: float = 70.0, show_images: bool = True):
    mu_img, masks, mu_tiss, px_cm = build_phantom(N, E_keV)
    sino, bp, fbp = ct_pipeline(mu_img)

    # mean μ in every region
    mean_mu = {t: mu_img[masks[t]].mean() for t in ["fat", "muscle", "tumor", "nerve"]}
    hu      = {t: hounsfield(mean_mu[t]) for t in mean_mu}

    # CNRs as requested
    cnr_to_muscle = {t: cnr(hu[t], hu["muscle"], N)
                     for t in ["fat", "tumor", "nerve"]}
    cnr_tumor_nerve = cnr(hu["tumor"], hu["nerve"], N)

    # tabulate
    df = (pd
          .DataFrame.from_dict(mean_mu, orient='index', columns=['μ (cm⁻¹)'])
          .assign(HU=[hu[t] for t in mean_mu])
          .assign(CNR_vs_muscle=[cnr_to_muscle.get(t, np.nan) for t in mean_mu])
          .rename_axis("Tissue")
          .round({"μ (cm⁻¹)": 5, "HU": 1, "CNR_vs_muscle": 2}))

    # tumour-to-nerve CNR separate line
    df.loc["tumor-to-nerve CNR"] = ["–", "–", round(cnr_tumor_nerve, 2)]

    # -----------------------------------------------------------------
    # visualisation
    # -----------------------------------------------------------------
    if show_images:
        # individual figures – one per the plotting rules
        for title, im, aspect in [
            ("Sinogram", sino, 'auto'),
            ("Unfiltered back-projection", bp, 'equal'),
            ("Filtered back-projection (Ram-Lak)", fbp, 'equal')]:
            plt.figure(figsize=(6, 6))
            plt.imshow(im, cmap='gray', aspect=aspect)
            plt.title(f"{title}  –  {N}×{N} px,  {E_keV} keV")
            plt.axis('off')
            plt.tight_layout()

        # phantom itself
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
    # choose 512, 1024 or 2048
    demo(N=512, E_keV=70.0)
