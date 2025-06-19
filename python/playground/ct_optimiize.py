import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from typing import Dict, Any

# ---------------------------------------------------------------------
# 0.  Parameters that describe the *real* object (all lengths in cm)
# ---------------------------------------------------------------------

tumor_radius      = 1.0                     # cm
muscle_thickness  = 3.0                     # cm
fat_thickness     = 0.5                     # cm
muscle_outer_r    = muscle_thickness        # 3.0 cm – tumour thickness removed
outer_radius      = muscle_outer_r + fat_thickness    # 3.5 cm
nerve_radius      = 0.5                     # cm
gap_to_tumor      = 0.05                    # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius  # 1.55 cm

# ---------------------------------------------------------------------
# 1.  Tissue dictionary & linear‑attenuation helper
# ---------------------------------------------------------------------

tissues: Dict[str, Dict[str, Any]] = {
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
# 2.  Noise helpers with bow-tie support
# ---------------------------------------------------------------------

EPS = 1e-6  # ε added to avoid log(0)

def add_poisson_noise(sino_px: np.ndarray,
                      I0: float | np.ndarray,
                      px_cm: float) -> np.ndarray:
    """Add quantum (shot) noise to a noiseless sinogram in pixel units.
       I0 may be a scalar or an array matching detector bins."""
    # convert back to physical line integral (cm⁻¹·cm)
    sino_cm = sino_px * px_cm
    # simulate photon counts (elementwise if I0 is array)
    expected = I0 * np.exp(-sino_cm)
    noisy    = np.random.poisson(lam=expected).astype(float)
    # avoid zeros
    noisy[noisy <= 0] = EPS
    # measured log-sinogram
    sino_cm_noisy = -np.log(noisy / I0)
    # convert back to pixel-unit sinogram
    return sino_cm_noisy / px_cm

# ---------------------------------------------------------------------
# 3.  CT pipeline with optional bow-tie filter
# ---------------------------------------------------------------------

def ct_pipeline(mu: np.ndarray,
                *,
                angles=None,
                filter_name: str = 'ramp',
                circle: bool = True,
                I0: float | None = None,
                px_cm: float = None,
                bowtie: bool = False,
                bowtie_strength: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-project *mu*, optionally apply bow-tie modulation, add Poisson noise, and reconstruct."""
    N = mu.shape[0]
    if mu.shape[1] != N:
        raise ValueError("mu image must be square")
    if angles is None:
        angles = np.linspace(0., 180., N, endpoint=False)

    # sinogram [projections x detector bins]
    sino = radon(mu, theta=angles, circle=circle)

    # apply noise if requested
    if I0 is not None:
        if bowtie:
            # detector bin centers in cm
            det_coords = (np.arange(sino.shape[1]) - sino.shape[1]/2 + 0.5) * px_cm
            # linear bow-tie profile: 1 at center, (1-strength) at edges
            profile = 1 - (np.abs(det_coords) / det_coords.max()) * bowtie_strength
            I0_profile = I0 * profile[np.newaxis, :]
        else:
            I0_profile = I0
        sino = add_poisson_noise(sino, I0_profile, px_cm)

    # backprojections
    backproj = iradon(sino, theta=angles, filter_name=None,
                      circle=circle, preserve_range=True)
    fbp      = iradon(sino, theta=angles, filter_name=filter_name,
                      circle=circle, preserve_range=True)
    return sino, backproj, fbp

# ---------------------------------------------------------------------
# 4.  Build a square phantom
# ---------------------------------------------------------------------

def build_phantom(N: int, E_keV: float = 70.0):
    field_cm = 2 * outer_radius
    px_cm    = field_cm / N
    coords   = (np.arange(N) - N/2 + 0.5) * px_cm
    X, Y     = np.meshgrid(coords, coords)
    R        = np.hypot(X, Y)
    mu_tiss  = {t: linear_attenuation(t, E_keV) for t in tissues}
    mu_img   = np.zeros((N, N), dtype=np.float32)

    # define masks
    fat_mask    = (R <= outer_radius) & (R > muscle_outer_r)
    tumor_mask  = (R <= tumor_radius)
    nerve_mask  = ((X + nerve_center_dist)**2 + Y**2) <= nerve_radius**2
    muscle_mask = (R <= muscle_outer_r) & (R > tumor_radius) & (~nerve_mask)

    mu_img[fat_mask]    = mu_tiss["fat"]
    mu_img[muscle_mask] = mu_tiss["muscle"]
    mu_img[tumor_mask]  = mu_tiss["tumor"]
    mu_img[nerve_mask]  = mu_tiss["nerve"]

    masks = {"fat": fat_mask, "muscle": muscle_mask,
             "tumor": tumor_mask, "nerve": nerve_mask}
    return mu_img, masks, mu_tiss, px_cm

# ---------------------------------------------------------------------
# 5.  HU conversion
# ---------------------------------------------------------------------

def mu_to_hu(mu_arr: np.ndarray, *, mu_water: float = 0.19) -> np.ndarray:
    return 1000.0 * (mu_arr - mu_water) / mu_water

# ---------------------------------------------------------------------
# 6.  Compute metrics for a single setting
# ---------------------------------------------------------------------

def compute_metrics(dose: float,
                    E_keV: float,
                    N: int,
                    bowtie: bool = False) -> Dict[str, Any]:
    # simulate
    mu_img, masks, mu_tiss, px_cm = build_phantom(N, E_keV)
    sino, bp, fbp = ct_pipeline(mu_img, I0=dose, px_cm=px_cm,
                                filter_name='ramp', bowtie=bowtie)

    # reconstruct and convert to HU
    hu_img = mu_to_hu(fbp)

    # mean HU per tissue
    mean_hu = {t: hu_img[masks[t]].mean() for t in masks}
    sigma_hu = hu_img[masks['muscle']].std(ddof=1)

    # HU-CNR vs muscle
    hu_cnr = {t: np.abs(mean_hu[t] - mean_hu['muscle']) / sigma_hu
              for t in ['fat', 'tumor', 'nerve']}
    # tumor-to-nerve
    hu_cnr_tn = np.abs(mean_hu['tumor'] - mean_hu['nerve']) / sigma_hu

    # contrast vs muscle
    contrast = {t: np.abs((mean_hu[t] - mean_hu['muscle']) / mean_hu['muscle'])
                for t in ['fat', 'tumor', 'nerve']}
    contrast_tn = np.abs((mean_hu['tumor'] - mean_hu['nerve']) / mean_hu['nerve'])

    return {
        'dose': dose,
        'E_keV': E_keV,
        'N': N,
        'bowtie': bowtie,
        'HU_CNR_vs_muscle': hu_cnr,
        'HU_CNR_tumor_nerve': hu_cnr_tn,
        'Contrast_vs_muscle': contrast,
        'Contrast_tumor_nerve': contrast_tn
    }

# ---------------------------------------------------------------------
# 7.  Example sweep and best settings selection
# ---------------------------------------------------------------------

def sweep_and_report(doses, energies, Ns, bowtie_opts):
    results = []
    for dose in doses:
        for E in energies:
            for N in Ns:
                for bt in bowtie_opts:
                    m = compute_metrics(dose, E, N, bt)
                    # define objective: minimal hu_cnr across tissues + tumor-nerve
                    obj_min = min(m['HU_CNR_vs_muscle'].values()) + m['HU_CNR_tumor_nerve']
                    m['objective'] = obj_min
                    results.append(m)
    df = pd.DataFrame(results)
    best = df.loc[df['objective'].idxmax()]
    print("Best settings based on objective:\n", best)
    return df, best

if __name__ == "__main__":
    # define parameter ranges
    doses   = [1e4, 1e5, 1e6, 1e7]
    energies= np.linspace(20, 150, 14)
    Ns      = [512, 1024, 2048]
    bowtie_opts = [False, True]

    df, best = sweep_and_report(doses, energies, Ns, bowtie_opts)
