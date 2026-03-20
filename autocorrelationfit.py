import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import signal  # Added for fast autocorrelation
from scipy.optimize import curve_fit  # Added for exponential fitting

# --- Configuration ---
PORT = "COM6"
BAUDRATE = 1000000
DURATION = 20  # seconds to record per run
NUM_RUNS = 10  # Number of times to repeat the recording

# Define the exponential fit function
def dls_fit(x, a, b, c):
    return a + b * np.exp(-c * x)

try:
    # 1. Initialize Serial Connection once
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    
    print(f"Connected to {PORT}. PRESS THE RESET BUTTON ON THE ARDUINO NOW!")
    time.sleep(2) # Gives the Arduino time to reset and clear the buffer [cite: 310]
    
    # Run the entire experiment multiple times
    for run in range(1, NUM_RUNS + 1):
        print(f"\n--- Starting Run {run} of {NUM_RUNS} ---")
        
        # Clear anything that piled up in the buffer while saving/plotting the last run
        ser.reset_input_buffer() 

        data_list = []
        print(f"Recording for {DURATION} seconds...")
        start_time = time.time()
        
        # 2. High-Speed Chunked Reading Loop
        while (time.time() - start_time) < DURATION:
            bytes_waiting = ser.in_waiting
            if bytes_waiting >= 2:
                # Ensure we only read an even number of bytes to maintain 16-bit alignment
                bytes_to_read = bytes_waiting - (bytes_waiting % 2) 
                raw = ser.read(bytes_to_read)
                
                # Convert raw bytes to 16-bit unsigned integers instantly
                chunk = np.frombuffer(raw, dtype=np.uint16)
                data_list.append(chunk)

        end_time = time.time()
        print("Recording finished.")

        # 3. Process the Data
        data = np.concatenate(data_list)
        
        # Calculate the actual time-base (dt)
        actual_duration = end_time - start_time
        num_samples = len(data)
        dt_microseconds = (actual_duration / num_samples) * 1e6
        dt_seconds = dt_microseconds * 1e-6 # Convert to seconds for the physics fit
        print(f"Collected {num_samples} samples.")
        print(f"Average time per sample (dt): {dt_microseconds:.2f} µs")

        # 4. Save Data with Timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"c:\\LilyMengDCF\\DLS_data_run{run}_{timestamp}.csv"
        np.savetxt(filename, data, delimiter=",", fmt='%d')
        print(f"Data saved to {filename}")

        # 5. Plot RAW Data to Diagnose "Noise"
        plt.figure(figsize=(10, 4))
        # Plotting only the first 2000 points to see the actual waveform shape
        plt.plot(data[:2000], linewidth=0.8, color='blue') 
        plt.title(f"Run {run}: Raw DLS Signal\nMean Voltage Level: {np.mean(data):.1f} (ADC value)")
        plt.xlabel("Sample Index")
        plt.ylabel("ADC Value (0 to 4095)")
        plt.tight_layout()
        
        # Save the plot and close it (so the loop doesn't pause!)
        raw_filename = f"c:\\LilyMengDCF\\Raw_Plot_run{run}_{timestamp}.png"
        plt.savefig(raw_filename)
        plt.close()

        # 6. Calculate Autocorrelation Function
        print("Calculating Autocorrelation...")
        x = np.asarray(data, dtype=float)
        x = x - np.mean(x)  # Subtract mean
        
        # Fast FFT-based correlation
        acf_full = signal.correlate(x, x, mode='full', method='fft')
        acf = acf_full[acf_full.size // 2:]  # Take only positive lags
        acf = acf / acf[0]  # Normalize so lag 0 is exactly 1.0

        # 7. Fit the Autocorrelation Curve
        plot_bound = min(4000, len(acf))  # Look at the first 4000 lags
        acf_to_fit = acf[:plot_bound]
        
        # Create an array of physical time lags in SECONDS for the x-axis
        time_lags = np.arange(plot_bound) * dt_seconds

        print("Fitting exponential curve...")
        try:
            # p0 provides an initial guess for [a, b, c]
            popt, pcov = curve_fit(dls_fit, time_lags, acf_to_fit, p0=(0.0, 1.0, 100.0))
            a, b, c = popt
            tau = 1 / c  # physical tau in seconds
            fit_success = True
        except RuntimeError:
            print("Fit failed to converge for this run! Plotting raw ACF only.")
            fit_success = False

        # 8. Plot Autocorrelation + Fit
        plt.figure(figsize=(8, 5))
        
        # Plot raw ACF dots
        plt.plot(time_lags, acf_to_fit, linestyle='None', marker='o', markersize=3, alpha=0.6, label="ACF Data")
        
        # Plot the fit line if it was successful
        if fit_success:
            plt.plot(time_lags, dls_fit(time_lags, *popt), color='crimson', linewidth=2, label="Exponential Fit")
            plt.axhline(a, color='gray', linestyle='--', linewidth=1, label="Asymptote $a$")
            
            # Annotation box
            textstr = (
                r"$f(x) = a + b e^{-cx}$" "\n"
                rf"$a = {a:.3e}$" "\n"
                rf"$b = {b:.3e}$" "\n"
                rf"$c = {c:.3e}$" "\n"
                rf"$\tau = 1/c = {tau:.3e}$ s"
            )
            plt.gca().text(
                0.95, 0.95, textstr,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
            )

        plt.title(f"Run {run}: Autocorrelation (Mean Subtracted)\ndt ~ {dt_microseconds:.2f} µs")
        plt.xlabel("Lag Time (Seconds)")
        plt.ylabel("Normalized ACF")
        plt.ylim(-0.2, 1.1)  # Fix y-axis to see the 0 to 1 decay clearly
        plt.xlim(0, time_lags[-1])
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        # Save the plot BEFORE closing
        acf_filename = f'c:\\LilyMengDCF\\DLS_Autocorrelation_run{run}_{timestamp}.png'
        plt.savefig(acf_filename)
        print(f"Plot saved as {acf_filename}")
        plt.close() # Close plot so the loop continues automatically

    print("\n--- All 10 runs completed successfully! ---")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Ensure serial port is closed
    if 'ser' in locals() and ser.is_open:
        ser.close()