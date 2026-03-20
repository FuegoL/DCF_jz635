import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, curve_fit, fsolve

# ==========================================
# 0. User Configuration & Physics Constants
# ==========================================
# Define your datasets here.
# "files": path to the .npz files for that specific sample
# "color": color theme for that sample's markers and fit line
# "q_min" / "q_max": the specific Brownian range for fitting (within 1.5–4.0 µm⁻¹)
datasets = {
    "Bulk": {
        "files": "/Volumes/MyMedia SSD/1.50/*.npz",   # UPDATE THIS PATH
        "color": "#1f77b4",            # Steel blue
        "marker": 'o',                  # Circle marker
        "q_min": 1.5,
        "q_max": 4.0
    },
    "Bottom": {
        "files": "/Volumes/MyMedia SSD/bottom/*.npz", # UPDATE THIS PATH
        "color": "#ff7f0e",              # Safety orange
        "marker": 's',                    # Square marker
        "q_min": 1.5,
        "q_max": 4.0
    }
}

# Stokes–Einstein constants
kb = 1.380649e-23      # Boltzmann constant [J/K]
T_kelvin = 298.15      # Temperature [K] (25°C)
eta = 8.9e-4           # Dynamic viscosity of water at 25°C [Pa·s]

# Particle radius for wall-distance calculation (from your known diameter)
DIAMETER = 3.0         # µm
r_particle = DIAMETER / 2.0

# ==========================================
# 1. Plotting Aesthetics
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'legend.fontsize': 15,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# ==========================================
# 2. Fitting Functions for DDM Data
# ==========================================
LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts / p[2])) + p[1], 1e-10))

def fit_single_isf(ISF_data, dts_data, tmax=None):
    """
    Fit each q‑mode of the image structure function (ISF) to a double‑exponential‑like model.
    Returns array of [amplitude, offset, tau_c] for every q.
    """
    params = np.zeros((ISF_data.shape[1], 3))
    for iq, ddm in enumerate(ISF_data.T):
        p0 = [np.ptp(ISF_data), ddm.min(), dts_data[len(dts_data)//2]]
        try:
            params[iq] = leastsq(
                lambda p, x, y: LogISF(p, x) - y,
                p0,
                args=(dts_data[:tmax] if tmax else dts_data,
                      np.log(np.maximum(ddm[:tmax] if tmax else ddm, 1e-10)))
            )[0]
        except:
            params[iq] = [np.nan, np.nan, np.nan]
    return params

def brownian_model(log_q, log_D):
    """Linear model: log(tau) = -log(D) - 2·log(q)"""
    return -log_D - 2.0 * log_q

# ==========================================
# 3. Main Processing Loop
# ==========================================
fig, ax = plt.subplots(figsize=(9, 7))
diffusion_coeffs = []   # Will store D [m²/s] for Bulk and Bottom

for label, config in datasets.items():
    file_paths = glob.glob(config["files"])
    if not file_paths:
        print(f"Warning: No files found for {label} at {config['files']}. Skipping.")
        continue

    print(f"Processing {label} ({len(file_paths)} files)...")

    # Load and fit all files for this sample
    all_params = []
    for file in file_paths:
        data = np.load(file)
        ISF, qs, dts = data['ISF'], data['qs'], data['dts']
        all_params.append(fit_single_isf(ISF, dts))

    all_params = np.array(all_params)          # shape (n_files, n_q, 3)
    tau_mean = np.nanmean(all_params[:, :, 2], axis=0)
    tau_std  = np.nanstd(all_params[:, :, 2], axis=0)

    # Select q‑range for the Brownian fit (as defined in config)
    fit_mask = (qs >= config["q_min"]) & (qs <= config["q_max"])
    q_valid = qs[fit_mask]
    tau_valid = tau_mean[fit_mask]
    tau_err   = tau_std[fit_mask]

    # Fit log(tau) = -log(D) - 2·log(q) to obtain D
    popt, pcov = curve_fit(brownian_model, np.log(q_valid), np.log(tau_valid), p0=[-1.0])
    log_D_fit = popt[0]
    log_D_err = np.sqrt(np.diag(pcov))[0]

    # Diffusion coefficient and uncertainty [m²/s]
    D_fit = np.exp(log_D_fit)
    D_err = D_fit * log_D_err
    diffusion_coeffs.append(D_fit)   # store for later wall‑distance calculation

    # (Optional) convert to µm²/s and compute particle diameter
    D_um2s = D_fit * 1e12
    diameter_m = (kb * T_kelvin) / (3 * np.pi * eta * D_fit)
    diameter_nm = diameter_m * 1e9
    print(f"  D = {D_um2s:.3f} µm²/s  →  diameter = {diameter_nm:.1f} nm")

    # ----- Plotting -----
    color = config["color"]
    marker = config["marker"]

    # Only plot q in the strict range 1.5–4.0 µm⁻¹
    plot_mask = (qs >= 1.5) & (qs <= 4.0) & (tau_mean > 0)
    q_plot = qs[plot_mask]
    tau_plot = tau_mean[plot_mask]
    tau_plot_err = tau_std[plot_mask]

    # 1. Fit line (over the plotted q range)
    tau_fit_line = 1 / (D_fit * q_plot**2)
    ax.plot(q_plot, tau_fit_line, color=color, linestyle='--', linewidth=2, zorder=2)

    # 2. All data points in [1.5, 4.0] (faint)
    ax.errorbar(q_plot, tau_plot, yerr=tau_plot_err,
                fmt='o', markerfacecolor='white', markeredgecolor=color,
                alpha=0.3, ecolor=color, capsize=2, elinewidth=1,
                markersize=5, zorder=1)

    # 3. Points actually used in the fit (bold)
    ax.errorbar(q_valid, tau_valid, yerr=tau_err,
                fmt='o', markerfacecolor='white', markeredgecolor=color,
                ecolor=color, capsize=4, elinewidth=1.5, markeredgewidth=1.5,
                markersize=8, label=label, zorder=3)

# ==========================================
# 4. Final Plot Formatting
# ==========================================
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1.5, 4.0)
ax.set_xlabel(r'Scattering Vector, $q \ (\mu m^{-1})$', fontweight='bold')
ax.set_ylabel(r'Characteristic Time, $\tau_c \ (s)$', fontweight='bold')
ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig('bottom_vs_bulk.png', dpi=300)
plt.show()

# ==========================================
# 5. Hydrodynamic Wall Distance (Faxén's Law)
# ==========================================
if len(diffusion_coeffs) == 2:
    D_bulk, D_bottom = diffusion_coeffs
    D_ratio = D_bottom / D_bulk

    def faxen_equation(x):
        """Faxén's law for parallel diffusion near a single wall:
           D_wall / D_bulk = 1 - (9/16)x + (1/8)x³ - (45/256)x⁴ - (1/16)x⁵
           where x = r / h  (r = particle radius, h = centre‑to‑wall distance).
        """
        return (1 - (9/16)*x + (1/8)*x**3 - (45/256)*x**4 - (1/16)*x**5) - D_ratio

    # Solve for x = r/h  (initial guess 0.5)
    x_sol = fsolve(faxen_equation, 0.5)[0]
    h_center = r_particle / x_sol          # centre‑to‑wall distance [µm]
    z_gap = h_center - r_particle           # surface‑to‑wall gap [µm]

    print("\n" + "="*50)
    print("          HYDRODYNAMIC WALL DISTANCE (Faxén's Law)")
    print("="*50)
    print(f"Particle radius (r)       : {r_particle:.3f} µm")
    print(f"Bulk diffusion coefficient: {D_bulk:.3e} m²/s")
    print(f"Bottom diffusion coeff.   : {D_bottom:.3e} m²/s")
    print(f"Diffusion ratio (γ)       : {D_ratio:.4f}")
    print("-"*50)
    print(f"Centre‑to‑wall distance (h): {h_center:.3f} µm")
    print(f"Surface‑to‑wall gap (z)    : {z_gap:.3f} µm")
    print("="*50)
else:
    print("\nWarning: Could not compute wall distance. Need both Bulk and Bottom datasets.")