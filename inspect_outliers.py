import pickle
import glob
import os
import numpy as np

base_dir = '/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113'
pkl_files = glob.glob(os.path.join(base_dir, '*', 'ego0', 'gnss_flow.pkl'))

MAX_VALID_SPEED = 60.0 # m/s (216 km/h)

print(f"Scanning for outliers (> {MAX_VALID_SPEED} m/s)...")

for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            
        keys = sorted(data.keys(), key=lambda x: int(x))
        if len(keys) < 2: continue
        
        translations = np.array([data[k]['translation'] for k in keys])
        speeds = np.linalg.norm(translations, axis=1) * 10
        
        # Check for outliers
        outlier_indices = np.where(speeds > MAX_VALID_SPEED)[0]
        
        if len(outlier_indices) > 0:
            print(f"\nFound {len(outlier_indices)} outliers in {pkl_file}")
            for idx in outlier_indices:
                frame_idx = keys[idx]
                speed_val = speeds[idx]
                speed_kmh = speed_val * 3.6
                print(f"  Frame {frame_idx}: Speed = {speed_val:.2f} m/s ({speed_kmh:.2f} km/h)")
                print(f"  Translation vector: {translations[idx]}")

    except Exception as e:
        print(f"Error processing {pkl_file}: {e}")
