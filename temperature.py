import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq, curve_fit

# ==========================================
# 0. User Configuration 
# ==========================================
# Define your temperature datasets here. 
# "T_celsius": The actual temperature for the D vs T plot
temperature_datasets = {
    "15 °C": {
        "T_celsius": 15.0,
        "files": "/Volumes/MyMedia SSD/temperature/15degreecelcious3_2026-02-27-185231-0000_ISF.npz",   # UPDATE THIS PATH
        "color": '#313695',
        "marker": "o",               
        "q_min": 1.5,
        "q_max": 4.0
    },
    "23 °C": {
        "T_celsius": 23.0,
        "files": "/Volumes/MyMedia SSD/1um100_100fps/LilyMeng_2026-02-26-184626-0000_ISF.npz",   # UPDATE THIS PATH
        "color": '#74add1',          # Purple
        "marker": "s",               
        "q_min": 1.5,                  
        "q_max": 4.0                   
    },
    "45 °C": {
        "T_celsius": 45.0,
        "files": "/Volumes/MyMedia SSD/temperature/50degreecelcious3_2026-02-27-184622-0000_ISF.npz",   # UPDATE THIS PATH
        "color": "#fee090",          # Red
        "marker": "^",               
        "q_min": 1.5,
        "q_max": 4.0
    },
        "65 °C": {
        "T_celsius": 65.0,
        "files": "/Volumes/MyMedia SSD/temperature/65degreecelcious_2026-02-27-182424-0000_ISF.npz",   # UPDATE THIS PATH
        "color": "#a50026",          # Red
        "marker": "^",               
        "q_min": 1.5,
        "q_max": 4.0
    }
}

# GLOBAL PLOTTING BOUNDS FOR TAU VS Q PLOT
Q_PLOT_MIN = 1.5
Q_PLOT_MAX = 4.0

# ==========================================
# 1. Plotting Aesthetics (Publishable Quality)
# ==========================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.edgecolor': 'black',
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 100,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})
# ==========================================
# 2. Define Fitting Functions 
# ==========================================
LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts / p[2])) + p[1], 1e-10))

def fit_single_isf(ISF_data, dts_data):
    params = np.zeros((ISF_data.shape[1], 3))
    for iq, ddm in enumerate(ISF_data.T):
        p0 = [np.ptp(ISF_data), ddm.min(), dts_data[len(dts_data)//2]]
        try:
            params[iq] = leastsq(
                lambda p, x, y: LogISF(p, x) - y,
                p0,
                args=(dts_data, np.log(np.maximum(ddm, 1e-10)))
            )[0]
        except:
            params[iq] = [np.nan, np.nan, np.nan]
    return params

# Updated Model: Power-Law Fit with free alpha
def powerlaw_model(log_q, log_D, alpha):
    return -log_D - alpha * log_q

# ==========================================
# 3. Main Processing Loop
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

temperatures = []
diffusion_coeffs = []
diffusion_errs = []

for label, config in temperature_datasets.items():
    file_paths = glob.glob(config["files"])
    if not file_paths:
        print(f"Warning: No files found for {label} at {config['files']}. Skipping.")
        continue
        
    print(f"Processing {label}...")
    
    all_params = []
    for file in file_paths:
        data = np.load(file)
        ISF, qs, dts = data['ISF'], data['qs'], data['dts']
        all_params.append(fit_single_isf(ISF, dts))
        
    all_params = np.array(all_params)  
    tau_mean = np.nanmean(all_params[:, :, 2], axis=0)
    tau_std = np.nanstd(all_params[:, :, 2], axis=0)

    # Filter strictly for the fit
    q_min, q_max = config["q_min"], config["q_max"]
    fit_mask = (qs >= q_min) & (qs <= q_max)
    
    q_valid = qs[fit_mask]
    tau_valid = tau_mean[fit_mask]
    tau_err = tau_std[fit_mask]

    # Power-Law Fit: log(tau_c) = -log(D) - alpha*log(q)
    # Initial guesses: log_D = -1.0, alpha = 2.0
    popt, pcov = curve_fit(powerlaw_model, np.log(q_valid), np.log(tau_valid), p0=[-1.0, 2.0])
    log_D_fit, alpha_fit = popt
    
    fit_errors = np.sqrt(np.diag(pcov))
    log_D_err = fit_errors[0]
    alpha_err = fit_errors[1]

    # Calculate Prefactor D and Error
    D_fit = np.exp(log_D_fit)
    D_err = D_fit * log_D_err  
    
    temperatures.append(config["T_celsius"])
    diffusion_coeffs.append(D_fit)
    diffusion_errs.append(D_err)

    # --- Plotting on Figure 1 (tau vs q) ---
    plot_mask = (qs >= Q_PLOT_MIN) & (qs <= Q_PLOT_MAX) & (tau_mean > 0)
    q_plot_all = qs[plot_mask]
    tau_plot_all = tau_mean[plot_mask]
    tau_plot_err = tau_std[plot_mask]

    color = config["color"]
    marker_shape = config["marker"] 
    
    # Fit Line using the specific extracted alpha
    tau_fit_line = 1 / (D_fit * q_plot_all**alpha_fit)
    ax1.plot(q_plot_all, tau_fit_line, color=color, linestyle='--', linewidth=2, zorder=2)
    
    # Excluded Data (Faint)
    ax1.errorbar(q_plot_all, tau_plot_all, yerr=tau_plot_err, 
                 fmt=marker_shape, markerfacecolor='white', markeredgecolor=color, 
                 alpha=0.3, ecolor=color, capsize=2, elinewidth=1, markersize=5, zorder=1)

    # Fitted Data (Bold) - Include alpha in the legend label dynamically!
    legend_label = rf'{label} ($\alpha={alpha_fit:.2f}$)'
    
    ax1.errorbar(q_valid, tau_valid, yerr=tau_err, 
                 fmt=marker_shape, markerfacecolor='white', markeredgecolor=color, 
                 ecolor=color, capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
                 markersize=8, label=legend_label, zorder=3)

# ==========================================
# 4. Final Formatting for Figure 1 (tau vs q)
# ==========================================
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(Q_PLOT_MIN, Q_PLOT_MAX)

ax1.set_xlabel(r'Scattering Vector, $q \ (\mu m^{-1})$', fontweight='bold')
ax1.set_ylabel(r'Characteristic Time, $\tau_c \ (s)$', fontweight='bold')

# Added a general title to the legend explaining the fit equation
ax1.legend(loc='upper right', title=r"Fit: $\tau_c \propto q^{-\alpha}$")

fig1.tight_layout()
#fig1.savefig('Temperature_tau_c_vs_q_alpha.png', dpi = 300)

# ==========================================
# 5. Generation of Figure 2 (D vs T)
# ==========================================
sort_idx = np.argsort(temperatures)
temperatures = np.array(temperatures)[sort_idx]
diffusion_coeffs = np.array(diffusion_coeffs)[sort_idx]
diffusion_errs = np.array(diffusion_errs)[sort_idx]

fig2, ax2 = plt.subplots(figsize=(7, 5))

ax2.errorbar(temperatures, diffusion_coeffs, yerr=diffusion_errs, 
             fmt='-o', color='black', markerfacecolor='#d62728', markeredgecolor='black',
             ecolor='black', capsize=5, elinewidth=2, markeredgewidth=1.5, 
             markersize=9, linewidth=1.5, zorder=3)

textstr = '\n'.join((
    r'$\bf{Trend\ Analysis:}$',
    r'Effective $D(T)$ increases due to:',
    r'1) Higher thermal energy ($k_B T$)',
    r'2) Lower solvent viscosity ($\eta$)'
))
props = dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, zorder=4)

ax2.set_xlabel(r'Temperature ($^\circ$C)', fontweight='bold')
ax2.set_ylabel(r'Effective Transport Prefactor, $D$', fontweight='bold')

x_padding = (max(temperatures) - min(temperatures)) * 0.1
if x_padding == 0: x_padding = 5
ax2.set_xlim(min(temperatures) - x_padding, max(temperatures) + x_padding)

fig2.tight_layout()
#fig2.savefig('Temperature_D_vs_T_alpha.png', dpi = 300)

plt.show()