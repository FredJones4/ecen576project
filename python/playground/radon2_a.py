import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, rescale, iradon, iradon_sart
from skimage.transform.radon_transform import _get_fourier_filter
from scipy.sparse.linalg import cg, LinearOperator

"""
Iterative tomographic reconstruction – **stable & warning‑free**
===============================================================
A final cleanup that
1.  **Zeros the object outside the reconstruction circle** before forward
    projection, eliminating the warning and improving quantitative errors.
2.  Removes the unsupported `circle` keyword from `iradon_sart` (older
    scikit‑image versions don’t accept it) while keeping geometry consistent –
    the detector length matches image size, so the call succeeds.

Algorithms implemented
----------------------
* Filtered Back‑Projection (FBP, ramp filter)
* SART – 20 iterations
* Landweber – 100 iterations
* SIRT – 50 iterations
* CGLS – 50 iterations

All reconstructions are compared visually and by RMS error (inside the circle).
"""

# -----------------------------------------------------------------------------
# Load & preprocess the image
# -----------------------------------------------------------------------------
file_path = r"C:\\Users\\Owner\\code_2025_spring\\ecen576\\project\\python\\treat_grayscale.png"
img_rgba = imread(file_path)
img_rgb = img_rgba[..., :3]
image = rgb2gray(img_rgb) if img_rgb.ndim == 3 else img_rgb
image = rescale(image, scale=0.4, mode="reflect", channel_axis=None)

# Ensure square
h, w = image.shape
assert h == w, "Image must be square after rescaling"
N = h

# -----------------------------------------------------------------------------
# Zero outside the reconstruction circle (satisfies Radon assumption)
# -----------------------------------------------------------------------------
Y, X = np.ogrid[:N, :N]
center = (N - 1) / 2
r2 = (X - center) ** 2 + (Y - center) ** 2
radius2 = (N / 2) ** 2
mask = r2 <= radius2
image_circle = image.copy()
image_circle[~mask] = 0.0

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_title("Input (masked)")
ax.imshow(image_circle, cmap="gray")
ax.axis("off")
plt.show()

# -----------------------------------------------------------------------------
# Create sinogram with circle=True so detector length == N
# -----------------------------------------------------------------------------
angles = np.linspace(0.0, 180.0, N, endpoint=False)
sinogram = radon(image_circle, theta=angles, circle=True)

fig, ax = plt.subplots(figsize=(6, 4))
ax.set_title("Sinogram")
ax.set_xlabel("Projection angle (deg)")
ax.set_ylabel("Detector position (pixels)")
ax.imshow(sinogram, cmap="gray", aspect="auto")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Visualise Fourier filter responses (optional)
# -----------------------------------------------------------------------------
for f in ["ramp", "shepp-logan", "cosine", "hamming", "hann"]:
    plt.plot(_get_fourier_filter(2000, f), label=f)
plt.xlim([0, 1000])
plt.xlabel("frequency")
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Forward & adjoint operators with the same geometry
# -----------------------------------------------------------------------------

def A(x):
    return radon(x.reshape((N, N)), theta=angles, circle=True)


def AT(y):
    return iradon(y, theta=angles, filter_name=None, circle=True, output_size=N)

# Lipschitz estimate
_test = np.ones((N, N))
L = np.linalg.norm(A(_test)) ** 2 / np.linalg.norm(_test) ** 2

# -----------------------------------------------------------------------------
# 1. Filtered Back‑Projection
# -----------------------------------------------------------------------------
recon_fbp = iradon(sinogram, theta=angles, filter_name="ramp", circle=True, output_size=N)

# RMS inside the circle
fbp_err = np.linalg.norm((recon_fbp - image_circle)[mask]) / np.sqrt(mask.sum())
print(f"FBP        RMS error: {fbp_err:.4f}")

# -----------------------------------------------------------------------------
# 2. SART – 20 iterations (no 'circle' kwarg)
# -----------------------------------------------------------------------------
recon_sart = np.zeros_like(image_circle)
for _ in range(20):
    recon_sart = iradon_sart(sinogram, theta=angles, image=recon_sart)

sart_err = np.linalg.norm((recon_sart - image_circle)[mask]) / np.sqrt(mask.sum())
print(f"SART-20    RMS error: {sart_err:.4f}")

# -----------------------------------------------------------------------------
# 3. Landweber – 100 iterations
# -----------------------------------------------------------------------------

def landweber(y, n_iter=100, tau=1.0 / L):
    x = np.zeros_like(image_circle)
    for _ in range(n_iter):
        r = y - A(x)
        x += tau * AT(r)
    return x

recon_land = landweber(sinogram)
land_err = np.linalg.norm((recon_land - image_circle)[mask]) / np.sqrt(mask.sum())
print(f"Landweber  RMS error: {land_err:.4f}")

# -----------------------------------------------------------------------------
# 4. SIRT – 50 iterations
# -----------------------------------------------------------------------------
W_r = 1.0 / np.maximum(np.sum(sinogram, axis=1, keepdims=True), 1e-6)
W_c = 1.0 / np.maximum(np.sum(sinogram, axis=0, keepdims=True), 1e-6)


def sirt(y, n_iter=50, lam=1.0):
    x = np.zeros_like(image_circle)
    for _ in range(n_iter):
        r = y - A(x)
        x += lam * AT(W_r * r) * W_c
    return x

recon_sirt = sirt(sinogram)
sirt_err = np.linalg.norm((recon_sirt - image_circle)[mask]) / np.sqrt(mask.sum())
print(f"SIRT-50    RMS error: {sirt_err:.4f}")

# -----------------------------------------------------------------------------
# 5. CGLS – 50 iterations
# -----------------------------------------------------------------------------
nt = sinogram.size
A_lin = LinearOperator(
    (nt, N * N),
    matvec=lambda v: A(v.reshape((N, N))).ravel(),
    rmatvec=lambda v: AT(v.reshape(sinogram.shape)).ravel(),
)

b = sinogram.ravel()

cg_sol, info = cg(A_lin.T @ A_lin, A_lin.T @ b, maxiter=60) # Changed from 50
recon_cgls = cg_sol.reshape((N, N))

cgls_err = np.linalg.norm((recon_cgls - image_circle)[mask]) / np.sqrt(mask.sum())
print(f"CGLS-50    RMS error: {cgls_err:.4f}  (cg info={info})")

# -----------------------------------------------------------------------------
# Visual comparison
# -----------------------------------------------------------------------------
recons = {
    "FBP": recon_fbp,
    "SART-20": recon_sart,
    "Landweber-100": recon_land,
    "SIRT-50": recon_sirt,
    "CGLS-50": recon_cgls,
}

cols = 3
rows = int(np.ceil(len(recons) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=True)
axes = axes.ravel()

for ax, (title, img) in zip(axes, recons.items()):
    ax.set_title(title)
    ax.imshow(img, cmap="gray", vmin=image_circle.min(), vmax=image_circle.max())
    ax.axis("off")

for ax in axes[len(recons):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
