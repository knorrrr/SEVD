import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

path = "/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113/train_info.pkl"
print(f"Loading {path}...")
with open(path, 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} frames")

speeds = []
yaw_rates = []

for item in data:
    t = item['translation'] # [x, y, z]
    r = item['rotation']    # [pitch, yaw, roll]
    
    speed = np.linalg.norm(t)
    yaw_rate = r[1]
    
    speeds.append(speed)
    yaw_rates.append(yaw_rate)

speeds = np.array(speeds)
yaw_rates = np.array(yaw_rates)

print(f"Speed stats: Min={speeds.min():.4f}, Max={speeds.max():.4f}, Mean={speeds.mean():.4f}, Median={np.median(speeds):.4f}")
print(f"Yaw rate stats: Min={yaw_rates.min():.4f}, Max={yaw_rates.max():.4f}, Mean={yaw_rates.mean():.4f}, Median={np.median(yaw_rates):.4f}")

# Abs yaw rate
abs_yaw = np.abs(yaw_rates)
print(f"Abs yaw rate P90: {np.percentile(abs_yaw, 90):.4f}")
print(f"Abs yaw rate P95: {np.percentile(abs_yaw, 95):.4f}")
print(f"Abs yaw rate P99: {np.percentile(abs_yaw, 99):.4f}")

# Histogram buckets
print("Yaw rate histogram (abs):")
hist, bins = np.histogram(abs_yaw, bins=[0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
for i in range(len(hist)):
    print(f"{bins[i]} - {bins[i+1]}: {hist[i]}")

print("Speed histogram:")
hist_s, bins_s = np.histogram(speeds, bins=[0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
for i in range(len(hist_s)):
    print(f"{bins_s[i]} - {bins_s[i+1]}: {hist_s[i]}")
