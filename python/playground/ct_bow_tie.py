import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from typing import Dict, Any

# ---------------------------------------------------------------------
# 0.  Phantom geometry parameters (all lengths in cm)
# ---------------------------------------------------------------------

tumor_radius      = 1.0                    # cm
muscle_thickness  = 3.0                    # cm
fat_thickness     = 0.5                    # cm
muscle_outer_r    = muscle_thickness       # 3.0 cm
outer_radius      = muscle_outer_r + fat_thickness  # 3.5 cm
nerve_radius      = 0.5                    # cm
gap_to_tumor      = 0.05                   # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius  # 1.55 cm

# ---------------------------------------------------------------------
# 1.  Tissue properties & attenuation helper
# ---------------------------------------------------------------------

tissues: Dict[str, Dict[str, float]] = {
    'fat':    {'xray_mu_linear': -0.0004, 'xray_mu_intercept': 0.196},
    'muscle': {'xray_mu_linear': -0.0065, 'xray_mu_intercept': 0.260},
    'tumor':  {'xray_mu_linear': -0.0008, 'xray_mu_intercept': 0.250},
    'nerve':  {'xray_mu_linear': -0.0065, 'xray_mu_intercept': 0.240},
}

def linear_attenuation(tissue: str, E_keV: float) -> float:
    '''Return μ (cm^-1) at a given energy.'''
    m = tissues[tissue]['xray_mu_linear']
    b = tissues[tissue]['xray_mu_intercept']
    return m * E_keV + b

# ---------------------------------------------------------------------
# 2.  Noise & bow-tie filter helpers
# ---------------------------------------------------------------------

EPS = 1e-6  # small value to avoid log(0)

def add_poisson_noise(sino_px: np.ndarray,
                      I0: float = 1e5,
                      px_cm: float = None,
                      filter_transmission: np.ndarray = None) -> np.ndarray:
    '''Add Poisson noise and apply optional bow-tie filter.'''
    sino_cm = sino_px * px_cm
    if filter_transmission is not None:
        I0_map = I0 * filter_transmission
    else:
        I0_map = I0
    expected = I0_map * np.exp(-sino_cm)
    noisy = np.random.poisson(lam=expected).astype(float)
    ratio = noisy / I0_map
    ratio[ratio <= 0] = EPS
    sino_cm_noisy = -np.log(ratio)
    return sino_cm_noisy / px_cm

def generate_bowtie_transmission(N: int,
                                 px_cm: float,
                                 max_thickness_cm: float,
                                 mu_filter: float) -> np.ndarray:
    '''Generate radial bow-tie filter transmission map.'''
    coords = (np.arange(N) - N/2 + 0.5) * px_cm
    r_norm = np.abs(coords) / coords.max()
    thickness = max_thickness_cm * r_norm
    transmission_1d = np.exp(-mu_filter * thickness)
    return np.tile(transmission_1d, (N, 1))

# ---------------------------------------------------------------------
# 3.  CT pipeline with optional bow-tie filter
# ---------------------------------------------------------------------

def ct_pipeline(mu: np.ndarray,
                angles=None,
                filter_name: str = 'ramp',
                circle: bool = True,
                I0: float | None = None,
                px_cm=None,
                filter_mu: float | None = None,
                filter_max_thickness: float | None = None):
    if mu.shape[0] != mu.shape[1]:
        raise ValueError('mu image must be square')
    N = mu.shape[0]
    if angles is None:
        angles = np.linspace(0., 180., N, endpoint=False)
    sino = radon(mu, theta=angles, circle=circle)
    if I0 is not None:
        if filter_mu is not None and filter_max_thickness is not None:
            transmission = generate_bowtie_transmission(
                N, px_cm, filter_max_thickness, filter_mu)
        else:
            transmission = None
        sino = add_poisson_noise(sino, I0, px_cm, filter_transmission=transmission)
    backproj = iradon(sino, theta=angles, filter_name=None,
                      circle=circle, preserve_range=True)
    fbp = iradon(sino, theta=angles, filter_name=filter_name,
                 circle=circle, preserve_range=True)
    return sino, backproj, fbp

# ---------------------------------------------------------------------
# 4.  Phantom construction
# ---------------------------------------------------------------------

def build_phantom(N: int, E_keV: float = 70.0):
    field_cm = 2 * outer_radius
    px_cm = field_cm / N
    coords = (np.arange(N) - N/2 + 0.5) * px_cm
    X, Y = np.meshgrid(coords, coords)
    R = np.hypot(X, Y)
    mu_tiss = {t: linear_attenuation(t, E_keV) for t in tissues}
    mu_img = np.zeros((N, N), dtype=np.float32)
    fat_mask = (R <= outer_radius) & (R > muscle_outer_r)
    tumor_mask = (R <= tumor_radius)
    nerve_mask = ((X + nerve_center_dist)**2 + Y**2) <= nerve_radius**2
    muscle_mask = (R <= muscle_outer_r) & (R > tumor_radius) & (~nerve_mask)
    mu_img[fat_mask] = mu_tiss['fat']
    mu_img[muscle_mask] = mu_tiss['muscle']
    mu_img[tumor_mask] = mu_tiss['tumor']
    mu_img[nerve_mask] = mu_tiss['nerve']
    masks = {'fat': fat_mask, 'muscle': muscle_mask,
             'tumor': tumor_mask, 'nerve': nerve_mask}
    return mu_img, masks, mu_tiss, px_cm

# ---------------------------------------------------------------------
# 5.  HU conversion
# ---------------------------------------------------------------------

def mu_to_hu(mu_arr: np.ndarray, *, mu_water: float = 0.19) -> np.ndarray:
    return 1000.0 * (mu_arr - mu_water) / mu_water

# ---------------------------------------------------------------------
# 6.  Main demo: metrics + visualization
# ---------------------------------------------------------------------

def demo(N: int = 512,
         E_keV: float = 70.0,
         I0: float | None = 5e4,
         show_images: bool = True,
         window_hu: int = 400,
         filter_mu: float | None = None,
         filter_max_thickness: float | None = None):
    mu_water = 0.19
    mu_img, masks, mu_tiss, px_cm = build_phantom(N, E_keV)
    sino, bp, fbp = ct_pipeline(mu_img,
                                 filter_name='ramp',
                                 I0=I0,
                                 px_cm=px_cm,
                                 filter_mu=filter_mu,
                                 filter_max_thickness=filter_max_thickness)
    hu_img = mu_to_hu(fbp, mu_water=mu_water)
    # compute mean HU and mu per tissue
    mean_hu = {t: hu_img[masks[t]].mean() for t in masks}
    mean_mu = {t: mu_img[masks[t]].mean() for t in masks}
    # noise and conversion
    sigma_hu = hu_img[masks['muscle']].std(ddof=1)
    sigma_mu = sigma_hu * mu_water / 1000.0
    # SNR
    snr = {t: mean_hu[t] / sigma_hu if sigma_hu else np.inf for t in mean_hu}
    # CNR vs muscle
    hu_cnr_to_muscle = {t: abs(mean_hu[t] - mean_hu['muscle']) / sigma_hu if sigma_hu else np.inf for t in ['fat','tumor','nerve']}
    mu_cnr_to_muscle = {t: abs(mean_mu[t] - mean_mu['muscle']) / sigma_mu if sigma_mu else np.inf for t in ['fat','tumor','nerve']}
    # tumor vs nerve
    hu_cnr_tn = abs(mean_hu['tumor'] - mean_hu['nerve']) / sigma_hu if sigma_hu else np.inf
    mu_cnr_tn = abs(mean_mu['tumor'] - mean_mu['nerve']) / sigma_mu if sigma_mu else np.inf
    contrast_to_muscle = {t: abs((mean_hu[t] - mean_hu['muscle']) / mean_hu['muscle']) for t in mean_hu}
    contrast_tn = abs((mean_hu['tumor'] - mean_hu['nerve']) / mean_hu['nerve'])
    # tabulate
    df = (pd.DataFrame.from_dict(mean_hu, orient='index', columns=['HU'])
          .assign(mu=[mean_mu[t] for t in mean_hu])
          .assign(SNR=[snr[t] for t in mean_hu])
          .assign(HU_CNR_vs_muscle=[hu_cnr_to_muscle.get(t, np.nan) for t in mean_hu])
          .assign(mu_CNR_vs_muscle=[mu_cnr_to_muscle.get(t, np.nan) for t in mean_hu])
          .assign(Contrast_vs_muscle=[contrast_to_muscle.get(t, np.nan) for t in mean_hu])
          .round({'HU':1,'mu':4,'SNR':2,'HU_CNR_vs_muscle':2,'mu_CNR_vs_muscle':2,'Contrast_vs_muscle':3})
          .rename_axis('Tissue'))
    df.loc['tumor-to-nerve HU_CNR'] = [np.nan, np.nan, np.nan, round(hu_cnr_tn,2), round(mu_cnr_tn,2), np.nan]
    df.loc['tumor-to-nerve mu_CNR'] = [np.nan, np.nan, np.nan, np.nan, round(mu_cnr_tn,2), np.nan]
    df.loc['tumor-to-nerve contrast'] = [np.nan, np.nan, np.nan, np.nan, np.nan, round(contrast_tn,3)]
    # print table
    print(df.to_string(float_format=lambda x: f'{x:.3f}'))
    # visualization
    if show_images:
        for title, im, aspect, clim in [
            ('Sinogram', sino, 'auto', None),
            ('Unfiltered backprojection', bp, 'equal', None),
            ('Filtered FBP (Ram-lak)', hu_img, 'equal', (-window_hu,window_hu))]:
            plt.figure(figsize=(6,6))
            plt.imshow(im, cmap='gray', aspect=aspect, vmin=(clim[0] if clim else None), vmax=(clim[1] if clim else None))
            plt.title(f'''{title} | N={N}, E_keV={E_keV}, I0={'∞' if I0 is None else I0}, Filter={'None' if filter_mu is None else f'mu={filter_mu}, t_max={filter_max_thickness}'}''')
            plt.axis('off'); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    userVariable = False
    for dose in [1e5,5e5,1e6,None]:
        print(f'''\n=== No filter: I0={dose} ===''')
        demo(N=512,E_keV=20.0,I0=dose, show_images=userVariable)
        print(f'''\n=== With bow-tie: I0={dose} ===''')
        demo(N=512,E_keV=20.0,I0=dose,filter_mu=0.22,filter_max_thickness=0.5, show_images=userVariable)
