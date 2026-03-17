import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.optimize import curve_fit

# --- 1. Experimental Data ---
# Angles in degrees and their uncertainties
angles_deg = np.array([29.7, 32.7, 36.1])
angles_err_deg = np.array([3.6, 3.4, 3.3])

# Relaxation times (tau) in seconds and their uncertainties
taus = np.array([92e-3, 57.2e-3, 38.9e-3])
tau_errors = np.array([15.3e-3, 15.4e-3, 7.2e-3])

# --- 2. Physical Constants ---
WAVELENGTH_NM = 632.8          # Laser wavelength in nm
WAVELENGTH_M = WAVELENGTH_NM * 1e-9 
REFRACTIVE_INDEX = 1.333       # Water at 20 C
TEMP_K = 293.15                # 20 C in Kelvin
VISCOSITY = 0.94e-3          # Viscosity of water at 20 C (Pa*s)
K_B = 1.380649e-23             # Boltzmann constant (J/K)

# --- 3. The Math (Physics & Error Propagation) ---
# Y-Axis: Gamma = 1 / tau
gammas = 1.0 / taus
gamma_errors = (1.0 / (taus**2)) * tau_errors 

# X-Axis: Scattering Vector Squared (q^2)
angles_rad = np.radians(angles_deg)
angles_err_rad = np.radians(angles_err_deg)

K_factor = (4 * np.pi * REFRACTIVE_INDEX) / WAVELENGTH_M
q = K_factor * np.sin(angles_rad / 2.0)
q_squared = q**2

# Error in q^2
q2_errors = (K_factor**2 / 2.0) * np.sin(angles_rad) * angles_err_rad

# --- Rigorous Weighted Linear Fit (Forced through origin) ---
def linear_fit(x, slope):
    return slope * x

# We use curve_fit with 'sigma' to weight the fit by your experimental errors!
popt, pcov = curve_fit(linear_fit, q_squared, gammas, sigma=gamma_errors, absolute_sigma=True)

measured_slope = popt[0]
slope_err = np.sqrt(pcov[0][0])

# Apply the DLS Factor of 2! (Measured Gamma = 2 * D * q^2)
D = measured_slope
D_err = slope_err

# Stokes-Einstein Equation (Diameter)
diameter_m = (K_B * TEMP_K) / (3 * np.pi * VISCOSITY * D)

# Propagate error to diameter: (Delta_d / d) = (Delta_D / D)
diameter_err_m = diameter_m * (D_err / D)

# Convert to micrometers
diameter_um = diameter_m * 1e6
diameter_err_um = diameter_err_m * 1e6

# --- 4. Publishable Plotting ---
plt.rcParams.update({
    'font.size': 12, 'axes.linewidth': 1.5, 'lines.linewidth': 2,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
})

fig, ax = plt.subplots(figsize=(7, 5), dpi=100)

# Plot the linear fit
q2_line = np.linspace(0, max(q_squared) * 1.1, 100)
ax.plot(q2_line, linear_fit(q2_line, measured_slope), color='#d62728', linestyle='--', zorder=1, 
        label=r"Weighted Fit ($\Gamma = D \cdot q^2$)")

# Plot the data points with BOTH x and y error bars
ax.errorbar(q_squared, gammas, xerr=q2_errors, yerr=gamma_errors, 
            fmt='o', color='#1f77b4', markersize=7, markeredgecolor='black', 
            capsize=4, capthick=1.5, zorder=2, label='Experimental Data')

# Formatting
ax.set_xlabel(r"Scattering Vector Squared, $q^2$ ($\mathrm{m^{-2}}$)", fontweight='bold')
ax.set_ylabel(r"Decay Rate, $\Gamma$ ($\mathrm{s^{-1}}$)", fontweight='bold')
ax.set_xlim(0, max(q_squared) * 1.1)
ax.set_ylim(0, max(gammas + gamma_errors) * 1.1)
ax.grid(True, linestyle='--', alpha=0.5)

# --- Rigorous LaTeX Formatting for Scientific Notation ---
# 1. Find the base-10 exponent of the main value (D)
exponent = int(np.floor(np.log10(D)))

# 2. Divide both D and D_err by 10^exponent to get their mantissas
D_mantissa = D / (10**exponent)
D_err_mantissa = D_err / (10**exponent)

# 3. Create the text string using proper LaTeX math formatting (\times 10^{exp})
textstr = (
    r"$\bf{Calculated\ Results:}$" "\n"
    rf"$D = ({D_mantissa:.2f} \pm {D_err_mantissa:.2f}) \times 10^{{{exponent}}} \ \mathrm{{m^2/s}}$" "\n"
    rf"$d_H = {diameter_um:.2f} \pm {diameter_err_um:.2f} \ \mathrm{{\mu m}}$"
)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle="square,pad=0.5", facecolor="#f8f9fa", edgecolor="black", alpha=0.9))

ax.legend(loc='lower right', framealpha=0.9, edgecolor="black")
plt.tight_layout()

plt.savefig("Gamma_vs_q2_Rigorous_2.0um.png", dpi=300)
plt.show()

print(f"True Diffusion Coefficient (D): {D:.3e} +/- {D_err:.1e} m^2/s")
print(f"Hydrodynamic Diameter (d_H): {diameter_um:.3f} +/- {diameter_err_um:.3f} μm")