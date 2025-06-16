import numpy as np
import random
import matplotlib.pyplot as plt
# ------------------------------------------------------------------
# Basic material description
# ------------------------------------------------------------------
class Material:
    def __init__(self, name, *, thickness, mu_a, ρ, c):
        """
        Parameters
        ----------
        thickness : float   # metres
        mu_a      : float   # dB / (MHz·m)
        ρ         : float   # kg / m³
        c         : float   # m / s
        """
        self.name = name
        self.Thickness = thickness
        self.MuA       = mu_a
        self.ρ         = ρ
        self.c         = c
        self.Z         = ρ * c          # acoustic impedance (Rayl)

# ------------------------------------------------------------------
# Example material library  (replace with real values as needed)
# ------------------------------------------------------------------
Fat    = Material('Fat',    thickness=0.010, mu_a= 0.5, ρ= 920, c=1478)
Muscle = Material('Muscle', thickness=0.030, mu_a= 1.0, ρ=1050, c=1540)
Tumor  = Material('Tumour', thickness=0.005, mu_a= 0.9, ρ=1030, c=1540)
Nerve  = Material('Nerve',  thickness=0.003, mu_a= 0.8, ρ=1040, c=1540)

material_list = [Fat, Muscle, Tumor, Nerve]

# ------------------------------------------------------------------
# Helper coefficients (normal incidence)
# ------------------------------------------------------------------
def transmission(m1: Material, m2: Material) -> float:
    """Amplitude transmission coefficient (normal incidence)."""
    return 2 * m1.Z / (m1.Z + m2.Z)

def reflection(m1: Material, m2: Material) -> float:
    """Magnitude of reflection coefficient (normal incidence)."""
    if m1 is m2:                         # same medium → no boundary
        return 1.0
    return abs(m2.Z - m1.Z) / (m2.Z + m1.Z)

# ------------------------------------------------------------------
# Random tumour–nerve gap   (2-5 mm → 0.002–0.005 m)
# ------------------------------------------------------------------
TN_gap = random.uniform(0.002, 0.005)

# ------------------------------------------------------------------
# Key depth boundaries inside the phantom
# ------------------------------------------------------------------
maxDepth = Fat.Thickness + Muscle.Thickness
b1 = Fat.Thickness
b2 = Fat.Thickness + Muscle.Thickness - Tumor.Thickness - TN_gap - Nerve.Thickness
b3 = Fat.Thickness + Muscle.Thickness - TN_gap - Nerve.Thickness
b4 = Fat.Thickness + Muscle.Thickness - Nerve.Thickness

# ------------------------------------------------------------------
# Spatial sampling
# ------------------------------------------------------------------
steps = 500                                # number of depth samples
z     = np.linspace(0.0, maxDepth, steps)  # depth axis (m)

# ------------------------------------------------------------------
# Frequency sampling  (MHz)
# ------------------------------------------------------------------
frequencies = np.linspace(3, 15, 5)        # e.g. 3, 6, 9, 12, 15 MHz
num_freqs   = frequencies.size
dataNames   = [f"{f:.1f} MHz" for f in frequencies]

# ------------------------------------------------------------------
# Speckle (scalar factor per depth, Rayleigh-distributed)
# ------------------------------------------------------------------
sigma_speckle = 1.0 / np.sqrt(np.pi / 2)
speckle       = np.random.rayleigh(scale=sigma_speckle, size=steps)

# ------------------------------------------------------------------
# Intensity initialisation
# ------------------------------------------------------------------
I0 = 1.0
Is = np.full((steps, num_freqs), I0)
matDepth = np.full(steps, Fat, dtype=object)   # <— new default fill

# ------------------------------------------------------------------
# Helper coefficients (normal incidence)
# ------------------------------------------------------------------
def reflection(m1, m2):
    if (m1 is None) or (m2 is None) or (m1 is m2):
        return 1.0
    return abs(m2.Z - m1.Z) / (m2.Z + m1.Z)


# ------------------------------------------------------------------
# Depth‐wise propagation loop
# ------------------------------------------------------------------
for i, d in enumerate(z):
    alpha   = np.zeros(num_freqs)            # running attenuation (dB)
    T_coeff = np.ones(num_freqs)             # cumulative transmission

    # ------------------------ FAT ------------------------
    if d > 0:
        d1     = min(d, Fat.Thickness)
        alpha += Fat.MuA * frequencies * d1 * 2         # two-way path
        matDepth[i] = Fat

    # ---------------------- MUSCLE 1 ---------------------
    if d > Fat.Thickness:
        T_coeff *= transmission(Fat, Muscle) * transmission(Muscle, Fat)
        d2      = min(d, b2) - Fat.Thickness
        alpha  += Muscle.MuA * frequencies * d2 * 2
        matDepth[i] = Muscle

    # ----------------------- TUMOUR ----------------------
    if d > b2:
        T_coeff *= transmission(Muscle, Tumor) * transmission(Tumor, Muscle)
        d3      = min(d, b3) - b2
        alpha  += Tumor.MuA * frequencies * d3 * 2
        matDepth[i] = Tumor

    # ---------------------- MUSCLE 2 ---------------------
    if d > b3:
        T_coeff *= transmission(Tumor, Muscle) * transmission(Muscle, Tumor)
        d4      = min(d, b4) - b3
        alpha  += Muscle.MuA * frequencies * d4 * 2
        matDepth[i] = Muscle

    # ----------------------- NERVE -----------------------
    if d > b4:
        T_coeff *= transmission(Muscle, Nerve) * transmission(Nerve, Muscle)
        d5      = d - b4
        alpha  += Nerve.MuA * frequencies * d5 * 2
        matDepth[i] = Nerve

    # ----------------- Intensity update -----------------
    if i > 0:
        Is[i] = (Is[i-1] *
                 T_coeff *
                 10.0 ** (-alpha / 10.0) *
                 reflection(matDepth[i-1], matDepth[i]))
    else:
        Is[i] = I0

# ------------------------------------------------------------------
# House-keeping: avoid log(0) and apply speckle
# ------------------------------------------------------------------
Is[Is == 0.0] = 1e-10
Is *= speckle[:, np.newaxis]                # broadcast along frequency axis

# ------------------------------------------------------------------
# Done — Is holds the simulated back-scatter intensity vs depth & freq
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Plot: Intensity vs. Depth (one curve per frequency)
# ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
for j, f in enumerate(frequencies):
    plt.plot(z * 1000, Is[:, j], label=f"{f:.1f} MHz")   # depth to mm

plt.xlabel("Depth (mm)")
plt.ylabel("Intensity (arb. units)")
plt.title("Simulated Back‑scatter Intensity vs Depth")
plt.legend()
plt.tight_layout()
plt.show()