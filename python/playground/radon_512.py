import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

def ct_pipeline(mu, *, angles=None, filter_name='ramp', circle=True):
    """
    Forward CT simulation and reconstruction.

    Parameters
    ----------
    mu : ndarray, shape (512, 512)
        Map of linear attenuation coefficients.
    angles : ndarray, optional
        Projection angles in degrees (default: 512 evenly-spaced views).
    filter_name : str or None
        FBP filter; set to None for plain back-projection.
    circle : bool
        Mask pixels outside the inscribed circle (standard CT convention).

    Returns
    -------
    sinogram : 2-D ndarray
    backproj : 2-D ndarray
    fbp      : 2-D ndarray
    """
    if mu.shape != (512, 512):
        raise ValueError("Input must be a 512×512 image")

    if angles is None:                       # 1 projection ≈ 1 detector row (rule of thumb)
        angles = np.linspace(0., 180., mu.shape[0], endpoint=False)   # degrees

    sinogram = radon(mu, theta=angles, circle=circle)                 # forward projection
    backproj = iradon(sinogram, theta=angles,
                      filter_name=None,       # ⇒ unfiltered BP
                      circle=circle,
                      preserve_range=True)

    fbp = iradon(sinogram, theta=angles,
                 filter_name=filter_name,     # ⇒ filtered BP (defaults to Ram-Lak)
                 circle=circle,
                 preserve_range=True)

    return sinogram, backproj, fbp

# --- demo with a Shepp-Logan phantom -----------------------------------------
if __name__ == "__main__":
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    phantom = resize(shepp_logan_phantom(), (512, 512),
                     mode='reflect', anti_aliasing=True)

    sino, bp, fbp = ct_pipeline(phantom)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["Sinogram", "Back-projection", "Filtered back-projection"]
    for a, im, t in zip(ax, [sino, bp, fbp], titles):
        a.imshow(im, cmap="gray", aspect='auto' if t=="Sinogram" else None)
        a.set_title(t); a.axis('off')
    plt.tight_layout(); plt.show()
