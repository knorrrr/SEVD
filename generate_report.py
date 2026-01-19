
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
        continue

    key_list = list(keys)
    if len(key_list) < 2:
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
        
        if v > MAX_VALID_SPEED:
            outliers += 1
            continue
            
        total_frames += 1
        
        # Classification
        if v < SPEED_STOP_THRESHOLD:
            cat = "Stopped"
        elif abs(w) < YAW_RATE_THRESHOLD:
            cat = "Straight"
        else:
            cat = "Curve"

        stats[cat]["count"] += 1
        stats[cat]["sum_speed"] += v
        
        # Speed Bins (km/h)
        v_kmh = v * 3.6
        if v < SPEED_STOP_THRESHOLD:
             speed_bins["0 km/h (Stop)"] += 1
        elif v_kmh < 10:
            speed_bins["0-10 km/h"] += 1
        elif v_kmh < 30:
            speed_bins["10-30 km/h"] += 1
        elif v_kmh < 50:
            speed_bins["30-50 km/h"] += 1
        elif v_kmh < 70:
            speed_bins["50-70 km/h"] += 1
        else:
            speed_bins["70+ km/h"] += 1

# Output
print("\n" + "="*40)
print("DATASET ANALYSIS REPORT")
print("="*40)
print(f"Total Frames Processed: {total_frames}")
print(f"Outliers Ignored (> {MAX_VALID_SPEED} m/s): {outliers}")
print("\n[MOTION TYPE CLASSIFICATION]")
print(f"{'Type':<15} | {'Count':<10} | {'Percent':<10} | {'Avg Speed (km/h)':<15}")
print("-" * 60)
for cat, data in stats.items():
    count = data["count"]
    pct = (count / total_frames * 100) if total_frames else 0
    avg_speed = (data["sum_speed"] / count * 3.6) if count else 0
    print(f"{cat:<15} | {count:<10} | {pct:>9.2f}% | {avg_speed:>15.2f}")

print("\n\n[SPEED DISTRIBUTION]")
print(f"{'Range':<15} | {'Count':<10} | {'Percent':<10}")
print("-" * 40)
for bins, count in speed_bins.items():
    pct = (count / total_frames * 100) if total_frames else 0
    print(f"{bins:<15} | {count:<10} | {pct:>9.2f}%")
print("="*40)
