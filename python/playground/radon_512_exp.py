"""
CT PIPELINE – forward Radon → back-projection → filtered back-projection
========================================================================
This reference implementation takes a 512 × 512 NumPy array *mu*
(linear attenuation coefficients, cm⁻¹) and returns:

    • sinogram       – forward Radon transform
    • backproj       – unfiltered back-projection
    • fbp           – Ram-Lak filtered back-projection

Bibliography (ordered as cited in the accompanying explanation)
----------------------------------------------------------------
[1]  scikit-image Radon example                           (https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html)
[2]  Radon transform – Wikipedia                          (https://en.wikipedia.org/wiki/Radon_transform)
[3]  Filtered back-projection in Python (GitHub)          (https://github.com/csheaff/filt-back-proj)
[4]  Ram-Lak filter derivation – Stack Overflow           (https://stackoverflow.com/q/6709871)
[5]  Sinogram coordinate interpretation – Stack Overflow  (https://stackoverflow.com/q/53366865)
[6]  PyTomography FBP notebook                            (https://pytomography.readthedocs.io/en/stable/notebooks/t_fbp.html)
[7]  Toupy filter implementation docs                     (https://toupy.readthedocs.io/en/latest/rst/toupy.tomo.html)
[8]  Ramp-filter scaling discussion – scikit-image #6174  (https://github.com/scikit-image/scikit-image/issues/6174)
[9]  Linear attenuation coefficient – Radiopaedia         (https://radiopaedia.org/articles/linear-attenuation-coefficient)
[10] Hounsfield scale – Wikipedia                         (https://en.wikipedia.org/wiki/Hounsfield_scale)
[11] Tomographic reconstruction – Wikipedia               (https://en.wikipedia.org/wiki/Tomographic_reconstruction)
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

def ct_pipeline(mu, *, angles=None, filter_name='ramp', circle=True):
    """
    Forward CT simulation and reconstruction.
    Explanation for how this converts correctly found here: https://chatgpt.com/share/e/684ee43e-4488-8001-9ab5-fb6e20f65d4e
    TODO: convert image lengths to proper pixel values
    Parameters
    ----------
    mu : ndarray, shape (512, 512)
        Linear attenuation coefficients (μ, cm⁻¹); cf. [9], [10].
    angles : ndarray, optional
        Projection angles in degrees (default: 512 evenly-spaced views).
    filter_name : str or None
        FBP filter (default 'ramp', i.e. Ram-Lak) – see [4], [7], [8].
    circle : bool
        Mask pixels outside the inscribed circle – standard CT convention.

    Returns
    -------
    sinogram : 2-D ndarray
    backproj : 2-D ndarray (no filter)
    fbp      : 2-D ndarray (filtered back-projection)
    """
    if mu.shape != (512, 512):
        raise ValueError("Input must be a 512×512 image of μ values")

    if angles is None:                       # ≈ one view per detector row
        angles = np.linspace(0., 180., mu.shape[0], endpoint=False)

    sinogram = radon(mu, theta=angles, circle=circle)          # forward
    backproj = iradon(sinogram, theta=angles, filter_name=None,
                      circle=circle, preserve_range=True)      # BP
    fbp      = iradon(sinogram, theta=angles, filter_name=filter_name,
                      circle=circle, preserve_range=True)      # FBP
    return sinogram, backproj, fbp


# --- demo --------------------------------------------------------------------
if __name__ == "__main__":
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = resize(shepp_logan_phantom(), (512, 512),
                     mode='reflect', anti_aliasing=True)

    sino, bp, fbp = ct_pipeline(phantom)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, im, title in zip(ax,
                            [sino, bp, fbp],
                            ["Sinogram", "Back-projection", "Filtered back-projection"]):
        a.imshow(im, cmap="gray", aspect='auto' if title=="Sinogram" else None)
        a.set_title(title); a.axis("off")
    plt.tight_layout(); plt.show()
