import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import radon, rescale, iradon, iradon_sart
from skimage.transform.radon_transform import _get_fourier_filter
import abracatabra  # pip install abracatabra[qt-pyside6]

# -----------------------------------------------------------------------------
# User parameters
# -----------------------------------------------------------------------------
file_path = r'C:\Users\Owner\code_2025_spring\ecen576\project\python\treat_grayscale.png'
num_iters = int(input("Enter number of SART iterations: "))

# -----------------------------------------------------------------------------
# Load and preprocess image
# -----------------------------------------------------------------------------
img_rgba = imread(file_path)
img_rgb  = img_rgba[..., :3]
image = img_rgb
if image.ndim == 3:
    image = rgb2gray(image)
# rescale if needed
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)

# Set up non-SART window
non_sart_win = abracatabra.TabbedPlotWindow(window_id='non_sart')

# Input image
fig = non_sart_win.add_figure_tab("Input")
ax = fig.add_subplot()
ax.set_title("Input cross‐section")
ax.imshow(image, cmap=plt.cm.Greys_r)
ax.axis('off')

# Sinogram
theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)
fig = non_sart_win.add_figure_tab("Sinogram")
ax = fig.add_subplot()
ax.set_title("Radon transform\n(Sinogram)")
ax.set_xlabel("Projection angle (deg)")
ax.set_ylabel("Projection position (pixels)")
dx = 0.5 * 180.0 / max(image.shape)
dy = 0.5 / sinogram.shape[0]
ax.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto',
)

# Fourier filters
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']
fig = non_sart_win.add_figure_tab("Fourier filters")
ax = fig.add_subplot()
ax.set_title("Fourier filter responses")
for f in filters:
    resp = _get_fourier_filter(2000, f)
    ax.plot(resp, label=f)
ax.set_xlim([0, 1000])
ax.set_xlabel('frequency')
ax.legend()

# FBP reconstruction
recon_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
fbp_err = recon_fbp - image
fig = non_sart_win.add_figure_tab("FBP Recon")
ax = fig.add_subplot()
ax.set_title("Reconstruction: Filtered back projection")
ax.imshow(recon_fbp, cmap=plt.cm.Greys_r)
ax.axis('off')

fig = non_sart_win.add_figure_tab("FBP Error")
ax = fig.add_subplot()
ax.set_title("FBP Reconstruction Error")
ax.imshow(fbp_err, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)
ax.axis('off')

non_sart_win.apply_tight_layout()

# Set up SART window
sart_win = abracatabra.TabbedPlotWindow(window_id='sart', ncols=2)
recon = None
for i in range(num_iters):
    if i == 0:
        recon = iradon_sart(sinogram, theta=theta)
    else:
        recon = iradon_sart(sinogram, theta=theta, image=recon)
    err = recon - image
    # Reconstruction tab
    fig = sart_win.add_figure_tab(f"SART iter {i+1}", col=0)
    ax = fig.add_subplot()
    ax.set_title(f"SART Reconstruction (iter {i+1})")
    ax.imshow(recon, cmap=plt.cm.Greys_r)
    ax.axis('off')
    # Error tab
    fig = sart_win.add_figure_tab(f"Error iter {i+1}", col=1)
    ax = fig.add_subplot()
    ax.set_title(f"Reconstruction Error (iter {i+1})")
    ax.imshow(err, cmap=plt.cm.Greys_r, vmin=-0.2, vmax=0.2)
    ax.axis('off')

sart_win.apply_tight_layout()

# Display all windows
abracatabra.show_all_windows()
