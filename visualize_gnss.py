import os
import re
import matplotlib.pyplot as plt
import glob

def visualize_gnss():
    base_dir = "/media/ssd/SEVD/carla/out/11_towns_each_6000_ticks_20251123_215345/"
    output_dir = "/media/ssd/SEVD/carla/out/11_towns_each_6000_ticks_20251123_215345/gnss_visualization"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Find all town directories
    town_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("Town")]
    town_dirs.sort()

    print(f"Found {len(town_dirs)} town directories.")

    for town_dir_name in town_dirs:
        town_path = os.path.join(base_dir, town_dir_name)
        gnss_file_path = os.path.join(town_path, "ego0", "gnss", "gnss.txt")

        if not os.path.exists(gnss_file_path):
            print(f"Warning: GNSS file not found for {town_dir_name} at {gnss_file_path}")
            continue

        lats = []
        lons = []

        print(f"Processing {town_dir_name}...")
        
        try:
            with open(gnss_file_path, 'r') as f:
                for line in f:
                    # Parse line like: GnssMeasurement(..., lat=-0.002249, lon=0.003013, ...)
                    match = re.search(r'lat=([-+]?\d*\.\d+|\d+),\s*lon=([-+]?\d*\.\d+|\d+)', line)
                    if match:
                        lats.append(float(match.group(1)))
                        lons.append(float(match.group(2)))
        except Exception as e:
            print(f"Error reading {gnss_file_path}: {e}")
            continue

        if not lats:
            print(f"No valid GNSS data found for {town_dir_name}")
            continue

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.plot(lons, lats, marker='.', linestyle='-', markersize=2, linewidth=0.5)
        plt.title(f"GNSS Trajectory - {town_dir_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.axis('equal') # Ensure aspect ratio is correct for a map

        output_file = os.path.join(output_dir, f"{town_dir_name}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    visualize_gnss()
