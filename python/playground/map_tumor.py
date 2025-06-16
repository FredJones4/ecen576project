import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def draw_tissue_grayscale(intensity_fat: float,
                          intensity_muscle: float,
                          intensity_tumor: float,
                          intensity_nerve: float,
                          figsize: tuple = (6, 6)):
    """
    Draws a cross-sectional tissue model in grayscale.

    Parameters:
    - intensity_fat: float between 0 (black) and 1 (white) for fat layer brightness
    - intensity_muscle: float between 0 (black) and 1 (white) for muscle layer brightness
    - intensity_tumor: float between 0 (black) and 1 (white) for tumor brightness
    - intensity_nerve: float between 0 (black) and 1 (white) for nerve brightness
    - figsize: figure size tuple (width, height)

    Returns:
    - fig, ax: Matplotlib figure and axes objects
    """
    # Scale: 1 unit = 1 cm
    tumor_radius = 1.0  # cm
    muscle_thickness = 3.0  # cm
    fat_thickness = 0.5  # cm
    nerve_radius = 0.5  # cm
    gap_to_tumor = 0.05  # cm

    muscle_outer_radius = tumor_radius + muscle_thickness
    outer_radius = muscle_outer_radius + fat_thickness
    nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw layers in grayscale
    fat = Circle((0, 0), outer_radius, color=(intensity_fat,)*3, zorder=1)
    muscle = Circle((0, 0), muscle_outer_radius, color=(intensity_muscle,)*3, zorder=2)
    tumor = Circle((0, 0), tumor_radius, color=(intensity_tumor,)*3, zorder=3)
    nerve = Circle((-nerve_center_dist, 0), nerve_radius, color=(intensity_nerve,)*3, zorder=4)

    for patch in (fat, muscle, tumor, nerve):
        ax.add_patch(patch)

    # Set limits
    lim = outer_radius + 1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    return fig, ax

# Example usage:
fig, ax = draw_tissue_grayscale(
    intensity_fat=0.9,
    intensity_muscle=0.6,
    intensity_tumor=0.3,
    intensity_nerve=0.1
)
plt.show()
