import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, rescale, iradon, iradon_sart
from skimage.transform.radon_transform import _get_fourier_filter

# ---------------------------------------------------------------------------
# Utility: show an image and (optionally) overlay numeric intensity values
# ---------------------------------------------------------------------------
def show_image_with_values(
        img,
        title='',
        step=1,
        fmt='{:.2f}',
        cmap=plt.cm.Greys_r):
    '''
    Display `img` and print pixel-intensity numbers on top.

    Parameters
    ----------
    img   : 2-D numpy array
    title : str  – figure title
    step  : int  – sample every `step` pixels (≥1) to avoid clutter
    fmt   : str  – format string for each number
    cmap  : matplotlib colormap
    '''
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title(title)
    ax.imshow(img, cmap=cmap)

    rows, cols = img.shape
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            val = fmt.format(img[i, j])
            txt_color = 'white' if img[i, j] < (img.max() * 0.5) else 'black'
            ax.text(j, i, val,
                    color=txt_color, ha='center', va='center', fontsize=6)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Load your custom image
# ---------------------------------------------------------------------------
file_path = r'C:/Users/Owner/code_2025_spring/ecen576/project/python/treat_grayscale.png'
img_rgba = imread(file_path)           # (H, W, 4)
img_rgb  = img_rgba[..., :3]           # keep R, G, B
image = rgb2gray(img_rgb)              # → float in [0, 1]

# Optional rescale so width ≈ #projection-angles
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

# Show the input slice *with* intensity values
auto_step = max(1, min(image.shape) // 32)   # ↓ density if big image
show_image_with_values(image,
                       title='Input cross-section (values overlaid)',
                       step=auto_step)

# ---------------------------------------------------------------------------
# Forward Radon transform (sinogram)
# ---------------------------------------------------------------------------
theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)

dx = 0.5 * 180.0 / max(image.shape)
dy = 0.5 / sinogram.shape[0]
fig, ax = plt.subplots(figsize=(5, 4))
ax.set_title('Radon transform\n(Sinogram)')
ax.set_xlabel('Projection angle (deg)')
ax.set_ylabel('Projection position (pixels)')
im = ax.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto')
plt.colorbar(im, ax=ax, label='Intensity')
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Plot Fourier filters for FBP
# ---------------------------------------------------------------------------
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
plt.figure(figsize=(6, 3))
for f in filters:
    resp = _get_fourier_filter(2000, f)
    plt.plot(resp, label=f)
plt.xlim([0, 1000])
plt.xlabel('frequency')
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Filtered Back Projection reconstruction
# ---------------------------------------------------------------------------
recon_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
fbp_err = np.sqrt(np.mean((recon_fbp - image)**2))
print(f'FBP rms reconstruction error: {fbp_err:.3g}')

vlim = dict(vmin=-0.2, vmax=0.2)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
a1.set_title('Reconstruction\nFiltered back projection')
a1.imshow(recon_fbp, cmap=plt.cm.Greys_r)
a2.set_title('Reconstruction error\nFiltered back projection')
a2.imshow(recon_fbp - image, cmap=plt.cm.Greys_r, **vlim)
plt.tight_layout()
plt.show()

# Overlay values on the FBP reconstruction
show_image_with_values(recon_fbp,
                       title='FBP reconstruction (values overlaid)',
                       step=auto_step)

# ---------------------------------------------------------------------------
# SART reconstruction (1 & 2 iterations)
# ---------------------------------------------------------------------------
recon_sart1 = iradon_sart(sinogram, theta=theta)
err1 = np.sqrt(np.mean((recon_sart1 - image)**2))
print(f'SART (1 iteration) rms reconstruction error: {err1:.3g}')

recon_sart2 = iradon_sart(sinogram, theta=theta, image=recon_sart1)
err2 = np.sqrt(np.mean((recon_sart2 - image)**2))
print(f'SART (2 iterations) rms reconstruction error: {err2:.3g}')

fig, axes = plt.subplots(2, 2, figsize=(8, 8.5), sharex=True, sharey=True)
axes = axes.ravel()
titles = ['Reconstruction\nSART',
          'Reconstruction error\nSART',
          'Reconstruction\nSART (2 iterations)',
          'Reconstruction error\nSART (2 iterations)']
imgs = [recon_sart1, recon_sart1 - image, recon_sart2, recon_sart2 - image]

for ax, title, img in zip(axes, titles, imgs):
    ax.set_title(title)
    ax.imshow(img, cmap=plt.cm.Greys_r,
              **(vlim if 'error' in title.lower() else {}))
plt.tight_layout()
plt.show()

# Overlay values on the final SART reconstruction
show_image_with_values(recon_sart2,
                       title='SART (2 iter) reconstruction (values overlaid)',
                       step=auto_step)
