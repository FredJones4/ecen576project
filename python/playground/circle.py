import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Scale: 1 unit = 1 cm
# Radii calculations
tumor_radius = 1  # cm
muscle_thickness = 3  # cm
fat_thickness = 0.5  # cm
muscle_outer_radius = tumor_radius + muscle_thickness
outer_radius = muscle_outer_radius + fat_thickness

nerve_radius = 0.5  # cm
gap_to_tumor = 0.05  # 0.5 mm in cm
nerve_center_dist = tumor_radius + gap_to_tumor + nerve_radius

# Create figure
fig, ax = plt.subplots(figsize=(6, 6))

# Draw fat layer
fat = Circle((0, 0), outer_radius, color='#FBE7A1', zorder=1, label='Fat')

# Draw muscle layer
muscle = Circle((0, 0), muscle_outer_radius, color='#E18D8D', zorder=2, label='Muscle')

# Draw tumor
tumor = Circle((0, 0), tumor_radius, color='#FFA07A', zorder=3, label='Tumor')

# Draw nerve (offset to the left)
nerve = Circle((-nerve_center_dist, 0), nerve_radius, color='#3CB4C4', zorder=4, label='Nerve')

# Add to plot
ax.add_patch(fat)
ax.add_patch(muscle)
ax.add_patch(tumor)
ax.add_patch(nerve)

# Annotations
# ax.text(-0.2, 0.2, 'Tumor', color='black', fontsize=12, zorder=5)
# ax.text(-nerve_center_dist - nerve_radius - 0.2, 0, 'Nerve', color='black', fontsize=12, zorder=5)
# ax.text(outer_radius * 0.6, 0, 'Muscle', color='black', fontsize=12, zorder=5)
# ax.text(outer_radius * 0.8, outer_radius * 0.8, 'Fat', color='black', fontsize=12, zorder=5)

# Configure plot
ax.set_xlim(-outer_radius - 1, outer_radius + 1)
ax.set_ylim(-outer_radius - 1, outer_radius + 1)
ax.set_aspect('equal')
ax.axis('off')

plt.show()
