# show_masks.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from typing import Dict, Any

# (unchanged tissue and geometry definitions)

tumor_radius      = 1.0                     # cm
muscle_thickness  = 3.0                     # cm
fat_thickness     = 0.5                     # cm
muscle_outer_r    = muscle_thickness        # 3.0 cm – tumour thickness removed
outer_radius      = muscle_outer_r + fat_thickness    # 3.5 cm
nerve_radius      = 0.5                     # cm
gap_to_tumor      = 0.05                    # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius  # 1.55 cm

def linear_attenuation(tissue: str, E_keV: float) -> float:
    """Return μ (cm⁻¹) at a given energy."""
    m = tissues[tissue]["xray_mu_linear"]
    b = tissues[tissue]["xray_mu_intercept"]
    return m * E_keV + b

tissues: Dict[str, Dict[str, Any]] = {
    "fat":    {"xray_mu_linear": -0.0004, "xray_mu_intercept": 0.196},
    "muscle": {"xray_mu_linear": -0.0065, "xray_mu_intercept": 0.260},
    "tumor":  {"xray_mu_linear": -0.0008, "xray_mu_intercept": 0.250},
    "nerve":  {"xray_mu_linear": -0.0065, "xray_mu_intercept": 0.240},
}

# build_phantom as per multi_ct_demo_poisson (with nerve excluded from muscle)
def build_phantom(N: int, E_keV: float = 70.0):
    field_cm = 2 * outer_radius                  # 7.0 cm – full body diameter
    px_cm    = field_cm / N

    coords = (np.arange(N) - N/2 + 0.5) * px_cm
    X, Y   = np.meshgrid(coords, coords)
    R      = np.hypot(X, Y)

    mu_tiss = {t: linear_attenuation(t, E_keV) for t in tissues}
    mu_img  = np.zeros((N, N), dtype=np.float32)

    fat_mask    = (R <= outer_radius) & (R > muscle_outer_r)
    muscle_mask = (R <= muscle_outer_r) & (R > tumor_radius)
    tumor_mask  = (R <= tumor_radius)
    nerve_mask  = ((X + nerve_center_dist)**2 + Y**2) <= nerve_radius**2

    # Exclude nerve region from muscle mask
    muscle_mask = muscle_mask & ~nerve_mask

    masks = {"fat": fat_mask, "muscle": muscle_mask,
             "tumor": tumor_mask, "nerve": nerve_mask}
    return mu_img, masks, mu_tiss, px_cm


def show_masks(N: int = 16, E_keV: float = 70.0):
    """
    Build a small phantom and print each tissue mask as a matrix of 0s and 1s,
    then print summaries of mask counts, coverage, overlaps, and the summed mask.
    """
    _, masks, _, _ = build_phantom(N, E_keV)

    total_pixels = N * N
    mask_sums = {name: int(mask.sum()) for name, mask in masks.items()}

    # Print each mask matrix
    for name, mask in masks.items():
        print(f"\n{name.upper()} mask ({N}×{N}):")
        mat = mask.astype(int)
        for row in mat:
            print(" ".join(str(v) for v in row))

    # Print summed mask matrix
    sum_mask = np.zeros((N, N), dtype=int)
    for mask in masks.values():
        sum_mask += mask.astype(int)
    print(f"\nSUMMED MASK (value = number of masks covering each pixel) ({N}×{N}):")
    for row in sum_mask:
        print(" ".join(str(v) for v in row))

    # Print mask pixel counts
    print("\nMask pixel counts:")
    for name, count in mask_sums.items():
        print(f"  {name}: {count}")

    # Compute coverage
    union_mask = sum_mask > 0
    covered = int(union_mask.sum())
    print(f"\nTotal pixels: {total_pixels}")
    print(f"Covered by any mask: {covered} pixels ({covered/total_pixels:.2%})")
    if covered < total_pixels:
        print(f"Uncovered pixels: {total_pixels - covered}")
    else:
        print("All pixels are covered by masks.")

    # Compute overlaps
    print("\nMask overlaps (pixels in multiple masks):")
    overlaps_found = False
    for i, m1 in enumerate(masks):
        for j, m2 in enumerate(masks):
            if j <= i:
                continue
            overlap = int((masks[m1] & masks[m2]).sum())
            if overlap > 0:
                overlaps_found = True
                print(f"  {m1} & {m2}: {overlap}")
    if not overlaps_found:
        print("  None.")


if __name__ == "__main__":
    show_masks()
