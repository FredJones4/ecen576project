import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, rescale, iradon, iradon_sart
from skimage.transform.radon_transform import _get_fourier_filter
from skimage.restoration import denoise_tv_chambolle
from scipy.sparse.linalg import cg, LinearOperator

"""
CT reconstruction playground – with regularised & accelerated solvers
=====================================================================
This version fixes the compatibility issue with `skimage.restoration.denoise_tv_chambolle`: newer
scikit‑image releases use `max_num_iter` instead of `n_iter_max`. The TV‑Landweber routine now calls

```python
 denoise_tv_chambolle(x, weight=tv_weight, max_num_iter=20, channel_axis=None)
```

Everything else (OS‑SART, Tikhonov CG, etc.) is unchanged.
"""

# -----------------------------------------------------------------------------
# 0. Load & preprocess the phantom
# -----------------------------------------------------------------------------
file_path = r"C:\\Users\\Owner\\code_2025_spring\\ecen576\\project\\python\\treat_grayscale.png"
img_rgba = imread(file_path)
img_rgb = img_rgba[..., :3]
img = rgb2gray(img_rgb) if img_rgb.ndim == 3 else img_rgb
img = rescale(img, scale=0.4, mode="reflect", channel_axis=None)

N = img.shape[0]
assert N == img.shape[1], "Image must be square after rescaling"

# Zero outside reconstruction circle
Y, X = np.ogrid[:N, :N]
center = (N - 1) / 2
circle = (X - center) ** 2 + (Y - center) ** 2 <= (N / 2) ** 2
img_circle = img.copy()
img_circle[~circle] = 0

plt.imshow(img_circle, cmap="gray")
plt.title("Input (masked)")
plt.axis("off")
plt.show()

# -----------------------------------------------------------------------------
# 1. Geometry & sinogram
# -----------------------------------------------------------------------------
angles = np.linspace(0.0, 180.0, N, endpoint=False)
sinogram = radon(img_circle, theta=angles, circle=True)

plt.imshow(sinogram, cmap="gray", aspect="auto")
plt.title("Sinogram")
plt.xlabel("Angle (deg)")
plt.ylabel("Detector")
plt.show()

# -----------------------------------------------------------------------------
# 2. Operators A and AT
# -----------------------------------------------------------------------------

def A(x):
    return radon(x.reshape((N, N)), theta=angles, circle=True)


def AT(y):
    return iradon(y, theta=angles, filter_name=None, circle=True, output_size=N)

# Lipschitz estimate for gradient steps
L = np.linalg.norm(A(np.ones((N, N)))) ** 2 / N ** 2

# Masked error helper
err = lambda rec: np.linalg.norm((rec - img_circle)[circle]) / np.sqrt(circle.sum())

# -----------------------------------------------------------------------------
# 3. FBP baseline
# -----------------------------------------------------------------------------
recon_fbp = iradon(sinogram, theta=angles, filter_name="ramp", circle=True, output_size=N)
print(f"FBP            RMS error: {err(recon_fbp):.4f}")

# -----------------------------------------------------------------------------
# 4. Vanilla SART (20 iters)
# -----------------------------------------------------------------------------
recon_sart = np.zeros_like(img_circle)
for _ in range(20):
    recon_sart = iradon_sart(sinogram, theta=angles, image=recon_sart)
print(f"SART-20        RMS error: {err(recon_sart):.4f}")

# -----------------------------------------------------------------------------
# 5. Ordered‑Subset SART (8×10)
# -----------------------------------------------------------------------------
recon_oss = np.zeros_like(img_circle)
S = 8
subsets = [np.arange(s, len(angles), S) for s in range(S)]
for _ in range(10):
    for idx in subsets:
        recon_oss = iradon_sart(sinogram[:, idx], theta=angles[idx], image=recon_oss)
print(f"OS-SART (80)   RMS error: {err(recon_oss):.4f}")

# -----------------------------------------------------------------------------
# 6. Landweber & TV‑Landweber
# -----------------------------------------------------------------------------

def landweber(y, n_iter=100, tau=1.0 / L):
    x = np.zeros_like(img_circle)
    for _ in range(n_iter):
        x += tau * AT(y - A(x))
    return x

recon_land = landweber(sinogram)
print(f"Landweber-100  RMS error: {err(recon_land):.4f}")

# TV‑regularised Landweber

def tv_landweber(y, n_iter=50, tau=1.0 / L, tv_weight=0.1):
    x = np.zeros_like(img_circle)
    for _ in range(n_iter):
        x += tau * AT(y - A(x))
        x = denoise_tv_chambolle(x, weight=tv_weight, max_num_iter=20, channel_axis=None)
    return x

recon_tv = tv_landweber(sinogram)
print(f"TV-Landweber   RMS error: {err(recon_tv):.4f}")

# -----------------------------------------------------------------------------
# 7. SIRT (50 iters)
# -----------------------------------------------------------------------------
W_r = 1.0 / np.maximum(np.sum(sinogram, axis=1, keepdims=True), 1e-6)
W_c = 1.0 / np.maximum(np.sum(sinogram, axis=0, keepdims=True), 1e-6)

def sirt(y, n_iter=50, lam=1.0):
    x = np.zeros_like(img_circle)
    for _ in range(n_iter):
        x += lam * AT(W_r * (y - A(x))) * W_c
    return x

recon_sirt = sirt(sinogram)
print(f"SIRT-50        RMS error: {err(recon_sirt):.4f}")

# -----------------------------------------------------------------------------
# 8. CGLS (50) & Tikhonov CG
# -----------------------------------------------------------------------------
nt = sinogram.size
A_lin = LinearOperator(
    (nt, N * N),
    matvec=lambda v: A(v.reshape(N, N)).ravel(),
    rmatvec=lambda v: AT(v.reshape(sinogram.shape)).ravel(),
)

b_vec = sinogram.ravel()

recon_cg_vec, info_cg = cg(A_lin.T @ A_lin, A_lin.T @ b_vec, maxiter=50)
recon_cg = recon_cg_vec.reshape(N, N)
print(f"CGLS-50        RMS error: {err(recon_cg):.4f}  (info={info_cg})")

# Tikhonov‑regularised CG

def tikh_cg(y, lam=1e-3, n_iter=50):
    rhs = AT(y).ravel()
    def mv(v):
        v_img = v.reshape(N, N)
        return (AT(A(v_img)) + lam * v_img).ravel()
    Areg = LinearOperator((N * N, N * N), matvec=mv)
    sol, info = cg(Areg, rhs, maxiter=n_iter)
    return sol.reshape(N, N), info

recon_tikh, info_tikh = tikh_cg(sinogram, lam=1e-3, n_iter=50)
print(f"Tikh-CG-50     RMS error: {err(recon_tikh):.4f}  (info={info_tikh})")

# -----------------------------------------------------------------------------
# 9. Montage of results
# -----------------------------------------------------------------------------
recons = {
    "FBP": recon_fbp,
    "SART-20": recon_sart,
    "OS-SART": recon_oss,
    "Landweber": recon_land,
    "TV-Landweber": recon_tv,
    "SIRT-50": recon_sirt,
    "CGLS-50": recon_cg,
    "Tikh-CG": recon_tikh,
}

cols = 3
rows = int(np.ceil(len(recons) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), sharex=True, sharey=True)
axes = axes.ravel()

vmin, vmax = img_circle.min(), img_circle.max()
for ax, (title, im) in zip(axes, recons.items()):
    ax.set_title(title)
    ax.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")

for ax in axes[len(recons):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
