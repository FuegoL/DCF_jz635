import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import leastsq, curve_fit

# ==========================================
# 0. User Configuration & Physics Constants
# ==========================================
# STRICT Q-RANGE FOR BROWNIAN FIT 
q_min_fit = 1.5  # Minimum q (um^-1)
q_max_fit = 4.0   # Maximum q (um^-1)

# Stokes-Einstein Constants
kb = 1.380649e-23      
T_kelvin = 295.15      
eta = 8.6e-4         

# ==========================================
# 1. Plotting Aesthetics (DLS-Aligned)
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

def fit_single_isf(ISF_data, dts_data, tmax=None):
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
    return -log_D - 2.0 * log_q

# ==========================================
# 3. Batch Loading and Processing
# ==========================================
file_paths = glob.glob('/Volumes/MyMedia SSD/3um100_100fps/*.npz') 
if not file_paths:
    raise ValueError("No .npz files found in the current directory.")

all_f_qt = []
all_params = []

for file in file_paths:
    data = np.load(file)
    ISF = data['ISF']
    qs = data['qs']
    dts = data['dts']
    
    params = fit_single_isf(ISF, dts, tmax=None)
    all_params.append(params)
    
    f_qt = np.zeros_like(ISF)
    for iq in range(len(qs)):
        A, B, tau_c = params[iq]
        f_qt[:, iq] = 1 - (ISF[:, iq] - B) / A
    all_f_qt.append(f_qt)

all_f_qt = np.array(all_f_qt)      
all_params = np.array(all_params)  

f_mean = np.nanmean(all_f_qt, axis=0)
f_std = np.nanstd(all_f_qt, axis=0)
tau_mean = np.nanmean(all_params[:, :, 2], axis=0)
tau_std = np.nanstd(all_params[:, :, 2], axis=0)

# ==========================================
# 4. Filter Data based on User Q-range
# ==========================================
fit_mask = (qs >= q_min_fit) & (qs <= q_max_fit)
valid_indices = np.where(fit_mask)[0]

q_valid = qs[fit_mask]
tau_valid = tau_mean[fit_mask]
tau_err = tau_std[fit_mask]

# ==========================================
# 5. Graph 1: Averaged f(q, tau) vs tau
# ==========================================
fig1, ax1 = plt.subplots(figsize=(8, 6))

# Strictly select 6 q values from INSIDE the valid fitting region
q_plot_indices = np.linspace(valid_indices[0], valid_indices[-1], 6, dtype=int)
colors = cm.viridis(np.linspace(0, 0.9, len(q_plot_indices)))

for idx, iq in enumerate(q_plot_indices):
    color = colors[idx]
    
    ax1.fill_between(dts, 
                     f_mean[:, iq] - f_std[:, iq], 
                     f_mean[:, iq] + f_std[:, iq], 
                     color='lightgrey', alpha=0.5, zorder=1)
    
    ax1.scatter(dts, f_mean[:, iq], 
                facecolors='none', edgecolors=color, s=35, zorder=2)
    
    fit_line = np.exp(-dts / tau_mean[iq])
    ax1.plot(dts, fit_line, color=color, linewidth=2, zorder=3,
             label=f'$q = {qs[iq]:.2f} \\,\\mu m^{{-1}}$')

ax1.set_xscale('log')
ax1.set_ylim(-0.05, 1.05)
ax1.set_xlim(dts[1]*0.8, dts[-1]*1.2)
ax1.set_xlabel(r'Lag Time $\tau$ (s)', fontweight='bold')
ax1.set_ylabel(r'Intermediate Scattering Function $f(q,\tau)$', fontweight='bold')
ax1.legend(loc='lower left', title="Brownian Region")
fig1.tight_layout()
fig1.savefig('f_q_tau_plot_3um.png')

# ==========================================
# 6. Graph 2: tau_c vs q Strict Brownian Fit
# ==========================================
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Fit log(tau_c) = -log(D) - 2*log(q)
popt, pcov = curve_fit(brownian_model, np.log(q_valid), np.log(tau_valid), p0=[-1.0])
log_D_fit = popt[0]
log_D_err = np.sqrt(np.diag(pcov))[0]

# Calculate D, errors, and Diameter
D_fit = np.exp(log_D_fit)
D_err = D_fit * log_D_err  

D_m2s = D_fit * 1e-12
D_m2s_err = D_err * 1e-12

diameter_m = (kb * T_kelvin) / (3 * np.pi * eta * D_m2s)
diameter_err_m = diameter_m * (D_err / D_fit)

diameter_nm = diameter_m * 1e9
diameter_err_nm = diameter_err_m * 1e9

# Background data mask for context
plot_mask = (tau_mean > 0)
q_plot_all = qs[plot_mask]
tau_plot_all = tau_mean[plot_mask]
tau_plot_err = tau_std[plot_mask]

# Plot full trend and strictly fitted line
tau_fit_line = 1 / (D_fit * q_plot_all**2)
ax2.plot(q_plot_all, tau_fit_line, color='#d62728', linestyle='--', linewidth=2.5, 
         label=r'Brownian Fit: $\tau_c = 1 / (D \cdot q^2)$', zorder=2)

ax2.errorbar(q_plot_all, tau_plot_all, yerr=tau_plot_err, 
             fmt='o', markerfacecolor='white', markeredgecolor='grey', 
             ecolor='lightgrey', capsize=3, elinewidth=1, markersize=5, 
             label='Excluded Data', zorder=1)

ax2.errorbar(q_valid, tau_valid, yerr=tau_err, 
             fmt='o', markerfacecolor='white', markeredgecolor='steelblue', 
             ecolor='grey', capsize=4, elinewidth=1.5, markeredgewidth=1.5, 
             markersize=8, label='Fitted Data', zorder=3)

# Results Box
textstr = '\n'.join((
    r'$\bf{Calculated\ Results:}$',
    rf'$D = {D_fit:.3f} \pm {D_err:.3f} \ \mu m^2/s$',
    rf'$d_h = {diameter_nm:.0f} \pm {diameter_err_nm:.0f} \ nm$'
))
props = dict(boxstyle='square,pad=0.6', facecolor='white', edgecolor='black', alpha=0.9)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=props, zorder=4)

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel(r'Scattering Vector, $q \ (\mu m^{-1})$', fontweight='bold')
ax2.set_ylabel(r'Characteristic Time, $\tau_c \ (s)$', fontweight='bold')

ax2.axvspan(q_min_fit, q_max_fit, color='steelblue', alpha=0.1, zorder=0, label='Fit Region')
ax2.legend(loc='lower left')

fig2.tight_layout()
fig2.savefig('tau_c_vs_q_strict_brownian_3um.png')
plt.show()