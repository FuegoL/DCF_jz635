import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d

# --- 1. Multi-Angle Configuration ---
# Define your angles and their corresponding folder paths here!
ANGLE_FOLDERS = {
    r"$\theta = 29.7\pm3.6^\circ$": "/Volumes/MyMedia SSD/LilyMengDCF/2.24_2um_angle3/*.csv",
    r"$\theta = 32.7\pm3.4^\circ$": "/Volumes/MyMedia SSD/LilyMengDCF/2.24_2um_angle2/*.csv",
    r"$\theta = 36.1\pm3.3^\circ$": "/Volumes/MyMedia SSD/LilyMengDCF/2.24_2um_angle1/*.csv",
    r"$\theta = 31.6\pm3.5^\circ$": "/Volumes/MyMedia SSD/LilyMengDCF/2.23 DLS angle2/*.csv",
    r"$\theta = 35.5\pm3.3^\circ$": "/Volumes/MyMedia SSD/LilyMengDCF/2.23 DLS Scatterangle1/*.csv",
    # Add as many as you need...
}

DT_MICROSECONDS = 72.16
DT_SECONDS = DT_MICROSECONDS * 1e-6
FIT_BOUND = 4000 
FS = 1.0 / DT_SECONDS 
Q = 5.0  

# --- Publishable Plotting Setup ---
plt.rcParams.update({
    'font.size': 12, 'axes.linewidth': 1.5, 'lines.linewidth': 2,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
})

def dls_fit(x, a, b, c):
    return a + b * np.exp(-c * x)

def preprocess_signal(x, window_size=30000, sigma=3.0):
    x = np.asarray(x, dtype=float)
    mean_val, std_val = np.mean(x), np.std(x)
    x_clean = np.clip(x, mean_val - sigma * std_val, mean_val + sigma * std_val)
    rolling_avg = uniform_filter1d(x_clean, size=window_size)
    return x_clean - rolling_avg

def fast_autocorrelation(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    acf_full = signal.correlate(x, x, mode='full', method='fft')
    acf = acf_full[acf_full.size // 2:]
    if acf[0] != 0: acf = acf / acf[0]
    return acf

# Initialize the main figure BEFORE the loop
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
# Generate a gradient of colors using the "viridis" or "plasma" colormap
import matplotlib.cm as cm

# Put this right before your loop
num_angles = len(ANGLE_FOLDERS)
colors = cm.viridis(np.linspace(0.1, 0.9, num_angles)) 

# (Note: 'viridis' goes from purple to green to yellow. 
# Other great academic options are 'plasma', 'cividis', or 'coolwarm')


# --- 2. Main Loop Over All Angles ---
for idx, (angle_name, folder_pattern) in enumerate(ANGLE_FOLDERS.items()):
    file_list = glob.glob(folder_pattern)
    if not file_list:
        print(f"Skipping {angle_name} - No files found at {folder_pattern}")
        continue

    print(f"Processing {angle_name}: {len(file_list)} datasets...")
    color = colors[idx % len(colors)] # Pick a color for this angle
    all_cleaned_acfs = []
    individual_taus = []
    
    # Process individual runs in this folder
    for file in file_list:
        data = np.loadtxt(file, delimiter=",")
        clean_data = preprocess_signal(data)
        acf = fast_autocorrelation(clean_data)
        
        plot_bound = min(FIT_BOUND, len(acf))
        acf_cut = acf[:plot_bound]
        time_lags = np.arange(plot_bound) * DT_SECONDS
        
        try:
            popt, _ = curve_fit(dls_fit, time_lags, acf_cut, p0=(0.0, 1.0, 100.0))
            residuals = acf_cut - dls_fit(time_lags, *popt)
            
            yf = np.fft.rfft(residuals)
            xf = np.fft.rfftfreq(len(residuals), d=DT_SECONDS)
            peaks, _ = signal.find_peaks(np.abs(yf), height=0.5, distance=20)
            
            acf_filtered = acf_cut.copy()
            for p in peaks:
                target_f = xf[p]
                if target_f > 80:
                    b, a = signal.iirnotch(target_f, Q, FS)
                    acf_filtered = signal.filtfilt(b, a, acf_filtered)
            
            all_cleaned_acfs.append(acf_filtered)
            popt_clean, _ = curve_fit(dls_fit, time_lags, acf_filtered, p0=popt)
            individual_taus.append(1 / popt_clean[2])
        except RuntimeError:
            pass

    # Calculate Master Curve for this angle
    if not all_cleaned_acfs:
        continue
        
    all_cleaned_acfs = np.array(all_cleaned_acfs)
    master_acf = np.mean(all_cleaned_acfs, axis=0)
    std_acf = np.std(all_cleaned_acfs, axis=0)
    tau_std = np.std(individual_taus) if individual_taus else 0.0
    # Fit Master Curve
    try:
        popt_master, _ = curve_fit(dls_fit, time_lags, master_acf, p0=(0.0, 1.0, 100.0))
        tau_global = 1 / popt_master[2]
    except RuntimeError:
        popt_master = [0, 0, 0]
        tau_global = 0

# --- 3. Add to Plot ---
    # Shadow (Error spread)
    ax.fill_between(time_lags, master_acf - std_acf, master_acf + std_acf, 
                    color='#888888', alpha=0.15, edgecolor='none')
    
    # Master Average (Hollow Scatter using the dynamic marker)
    ax.plot(time_lags, master_acf, markersize=1, 
            markerfacecolor='none', markeredgecolor='#888888', marker='o',alpha=0.1,
            linestyle='None', zorder=3)
    
    # Global Fit (Thinner line using the dynamic line style)
    ax.plot(time_lags, dls_fit(time_lags, *popt_master), 
            linewidth=1.5, linestyle= '-',color=color,label=f"{angle_name} ($\\tau$ = {tau_global * 1000:.1f} $\\pm$ {tau_std * 1000:.1f} ms)",zorder=4)

# --- 4. Final Plot Formatting ---
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle='--', alpha=0.5)

ax.set_xlabel("Lag Time (Seconds)")
ax.set_ylabel(r"$g^{(2)}(\tau) - 1$", fontweight='bold', fontsize=12)

# 1. Change to Log Scale
ax.set_xscale('log')

# 2. Start the x-axis at the first non-zero time lag so log() doesn't break
ax.set_xlim(time_lags[1], time_lags[-1])

ax.legend(loc='lower left', framealpha=0.9, edgecolor="black", fontsize=10)
plt.tight_layout()

#plt.savefig("DLS_2um_Scurve.png", dpi=300)
plt.show()