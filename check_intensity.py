
import numpy as np
import os

def check_intensity(path):
    print(f"Checking: {path}")
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    intensity = data[:, 3]
    print(f"  Count: {len(intensity)}")
    print(f"  Min: {intensity.min()}")
    print(f"  Max: {intensity.max()}")
    print(f"  Mean: {intensity.mean()}")
    print(f"  Std: {intensity.std()}")
    print("-" * 20)

base_dir = "/media/ssd/SEVD/carla/out/11_towns_each_6000_ticks_20251123_215345/Town01_Opt_23_11_2025_21_54_11/ego0"
file_name = "0003650.bin"

path1 = os.path.join(base_dir, "lidar-front_filtered", file_name)
path2 = os.path.join(base_dir, "lidar-front_filtered_downsampled", file_name)

check_intensity(path1)
check_intensity(path2)
