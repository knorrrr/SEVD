
import glob
import pickle
import numpy as np
import os

base_dir = '/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113'
pkl_files = glob.glob(os.path.join(base_dir, '*', 'ego0', 'gnss_flow.pkl'))

# Thresholds
SPEED_STOP_THRESHOLD = 0.1  # m/s
YAW_RATE_THRESHOLD = 2.0    # deg/s
MAX_VALID_SPEED = 60.0      # m/s (~216 km/h) - filter outliers

# Categories
stats = {
    "Stopped": {"count": 0, "sum_speed": 0},
    "Straight": {"count": 0, "sum_speed": 0},
    "Curve": {"count": 0, "sum_speed": 0}
}

speed_bins = {
    "0 km/h (Stop)": 0,
    "0-10 km/h": 0,
    "10-30 km/h": 0,
    "30-50 km/h": 0,
    "50-70 km/h": 0,
    "70+ km/h": 0
}

# Initialize Matrix
matrix = {
    "Stopped": {k: 0 for k in speed_bins.keys()},
    "Straight": {k: 0 for k in speed_bins.keys()},
    "Curve": {k: 0 for k in speed_bins.keys()}
}

total_frames = 0
outliers = 0

for pkl_file in pkl_files:
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {pkl_file}: {e}")
        continue
    
    # Sort keys
    try:
        keys = sorted(data.keys(), key=lambda x: int(x))
    except:
        print(f"Skipping file due to sort error: {pkl_file}")
        continue

    key_list = list(keys)
    if len(key_list) < 2:
        print(f"Skipping file due to <2 frames: {pkl_file}")
        continue
    
    # Extract arrays
    translations = np.array([data[k]['translation'] for k in key_list])
    rotations = np.array([data[k]['rotation'] for k in key_list])
    
    # Speed (m/s)
    frame_speeds = np.linalg.norm(translations, axis=1) * 10
    
    # Yaw Rate (deg/s)
    # rotations[:, 1] is already delta per frame in the pickle
    frame_yaw_rates = rotations[:, 1] * 10
    
    for i in range(len(key_list)):
        v = frame_speeds[i]
        w = frame_yaw_rates[i]
        
        # if v > MAX_VALID_SPEED:
        #     outliers += 1
        #     continue
            
        total_frames += 1
        
        # Classification (Motion)
        if v < SPEED_STOP_THRESHOLD:
            motion_cat = "Stopped"
        elif abs(w) < YAW_RATE_THRESHOLD:
            motion_cat = "Straight"
        else:
            motion_cat = "Curve"

        # Classification (Speed Bin)
        v_kmh = v * 3.6
        if v < SPEED_STOP_THRESHOLD:
             speed_cat = "0 km/h (Stop)"
        elif v_kmh < 10:
            speed_cat = "0-10 km/h"
        elif v_kmh < 30:
            speed_cat = "10-30 km/h"
        elif v_kmh < 50:
            speed_cat = "30-50 km/h"
        elif v_kmh < 70:
            speed_cat = "50-70 km/h"
        else:
            speed_cat = "70+ km/h"
            
        matrix[motion_cat][speed_cat] += 1

# Output
print("\n" + "="*80)
print("DATASET ANALYSIS REPORT (MATRIX)")
print("="*80)
print(f"Total Frames: {total_frames}")
print(f"Outliers Ignored (> {MAX_VALID_SPEED} m/s): {outliers}")
print("-" * 80)
print(f"Definitions:")
print('  "stopped":  speed < 0.1 m/s')
print('  "straight": speed >= 0.1 m/s AND abs(yaw_rate) < 2.0 deg/s')
print('  "curve":    speed >= 0.1 m/s AND abs(yaw_rate) >= 2.0 deg/s')
print("-" * 80)

# Print Matrix Header
headers = list(speed_bins.keys())
header_str = f"{'Motion':<10} | " + " | ".join([f"{h:<13}" for h in headers]) + " | Total"
print(header_str)
print("-" * len(header_str))

# Print Rows
for motion in ["Stopped", "Straight", "Curve"]:
    row_counts = [matrix[motion][h] for h in headers]
    row_total = sum(row_counts)
    row_str = f"{motion:<10} | " + " | ".join([f"{c:<13}" for c in row_counts]) + f" | {row_total}"
    print(row_str)

print("-" * len(header_str))

# Print Column Totals
col_totals = [sum(matrix[m][h] for m in ["Stopped", "Straight", "Curve"]) for h in headers]
grand_total = sum(col_totals)
total_str = f"{'TOTAL':<10} | " + " | ".join([f"{c:<13}" for c in col_totals]) + f" | {grand_total}"
print(total_str)
print("="*80)
