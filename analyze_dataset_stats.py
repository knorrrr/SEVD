
import glob
import pickle
import numpy as np
import os

base_dir = '/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113'
pkl_files = glob.glob(os.path.join(base_dir, '*', 'ego0', 'gnss_flow.pkl'))

print(f"Found {len(pkl_files)} files.")

speeds = []
yaw_rates = []

for pkl_file in pkl_files:
    print(f"Processing {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Sort keys
    try:
        keys = sorted(data.keys(), key=lambda x: int(x))
    except:
        print(f"Skipping {pkl_file}: keys not convertible to int")
        continue

    # Convert to list to access by index
    key_list = list(keys)
    
    # We iterate and compute stats
    # Optimization: processing in batches or list comprehension
    
    # Extract arrays
    translations = np.array([data[k]['translation'] for k in key_list])
    rotations = np.array([data[k]['rotation'] for k in key_list])
    
    # Calculate Speeds (m/s)
    # Norm of translation * 10
    file_speeds = np.linalg.norm(translations, axis=1) * 10
    speeds.extend(file_speeds[1:]) # Skip first if aligning with yaw rates? 
    # Actually translation[i] corresponds to movement from i-1 to i? 
    # Or i to i+1? Usually flow at frame t represents t -> t+1 or t-1 -> t.
    # Assuming it corresponds to the frame key, it's valid for that frame.
    
    # Calculate Yaw Rates (deg/s)
    # We need diff.
    # Index 1 is Yaw
    yaws = rotations[:, 1]
    yaw_diffs = yaws[1:] - yaws[:-1]
    
    # Wrap handling
    yaw_diffs = (yaw_diffs + 180) % 360 - 180
    
    file_yaw_rates = yaw_diffs * 10
    yaw_rates.extend(file_yaw_rates)
    
    # Note: speeds should be aligned. 
    # If using speeds[1:], we match yaw_rates size.
    # We'll use speeds[1:] to correspond to the interval `t-1` to `t`.
    
speeds = np.array(speeds)
yaw_rates = np.array(yaw_rates)

print("\n--- Statistics ---")
print(f"Total frames: {len(speeds)}")
print(f"Speed (m/s) - Mean: {np.mean(speeds):.2f}, Max: {np.max(speeds):.2f}, Min: {np.min(speeds):.2f}")
print(f"Yaw Rate (deg/s) - Mean: {np.mean(np.abs(yaw_rates)):.2f}, Max: {np.max(np.abs(yaw_rates)):.2f}")

# Threshold Analysis
print("\nPercentiles:")
print("Speed 10th:", np.percentile(speeds, 10))
print("Speed 50th:", np.percentile(speeds, 50))
print("Speed 90th:", np.percentile(speeds, 90))

print("Yaw Rate 50th:", np.percentile(np.abs(yaw_rates), 50))
print("Yaw Rate 90th:", np.percentile(np.abs(yaw_rates), 90))
print("Yaw Rate 99th:", np.percentile(np.abs(yaw_rates), 99))
