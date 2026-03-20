import os
import glob
import numpy as np

def correct_time_lapse_with_known_fps(directory, real_time_interval_seconds, original_fps):
    """
    Reverse-engineers the integer frame gaps using a KNOWN arbitrary framerate,
    then recalculates the true physical time axis.
    """
    
    # Find all npz files in the directory
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    
    if not npz_files:
        print("No .npz files found in the specified directory.")
        return

    print(f"Found {len(npz_files)} files. Applying correction using known fps...")
    
    # The true frames-per-second is 1 divided by your physical interval
    true_fps = 1.0 / real_time_interval_seconds

    for file in npz_files:
        with np.load(file) as data:
            data_dict = dict(data)
            
            # 1. Extract the incorrect times
            if 'dts' not in data_dict:
                print(f"Skipping {os.path.basename(file)}: 'dts' array not found.")
                continue
                
            bad_dts = data_dict['dts']
            
            # 2. REVERSE ENGINEER: Reconstruct the exact integer frame gaps
            # We multiply by the original_fps you provide, then snap to integers
            idts = np.round(bad_dts * original_fps).astype(int)
            
            # 3. Calculate the true physical lag times
            true_dts = idts * real_time_interval_seconds
            
            # 4. Update the dictionary with the correct physical values
            data_dict['dts'] = true_dts
            data_dict['fps'] = true_fps # We save the correct fps in case you need it later
            
            # 5. Save safely as a new file to prevent overwriting originals
            base_name, ext = os.path.splitext(file)
            new_file_name = f"{base_name}_corrected{ext}"
            
            np.savez_compressed(new_file_name, **data_dict)
            print(f"Corrected and saved: {os.path.basename(new_file_name)}")
            
    print("\nAll files successfully reverse-engineered and corrected!")

# ==========================================
# How to use the script:
# ==========================================
if __name__ == "__main__":
    # Point this to the folder containing your bad .npz files
    target_folder = r'/Volumes/MyMedia SSD/peopeg_mixture/28/500ms'
    
    # Set your true experimental interval in seconds (e.g., 500 ms = 0.5 seconds)
    actual_dt = 0.5 
    
    # Set the wrong framerate that the video was originally processed with.
    # If OpenCV processed an .avi without explicit timestamps, it often defaults to 30.
    known_bad_fps = 100 
    
    correct_time_lapse_with_known_fps(target_folder, actual_dt, known_bad_fps)