import os
import re
import matplotlib.pyplot as plt
import glob

def visualize_gnss():
    base_dir = "/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113"
    output_dir = "/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113/gnss_visualization"
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

        timestamps = []
        lats = []
        lons = []
        xs = []
        ys = []
        zs = []

        print(f"Processing {town_dir_name}...")
        
        try:
            with open(gnss_file_path, 'r') as f:
                for line in f:
                    # Example line: 
                    # GnssMeasurement(frame=3650, timestamp=2.981468, lat=-0.002242, lon=0.003014, alt=1.735297), Transform(Location(x=335.489319, y=249.617813, z=1.735297), Rotation(pitch=0.011693, yaw=90.029022, roll=-0.004456))
                    
                    # Extract timestamp, lat, lon
                    match_gnss = re.search(r'timestamp=([-+]?\d*\.\d+|\d+).*?lat=([-+]?\d*\.\d+|\d+),\s*lon=([-+]?\d*\.\d+|\d+)', line)
                    # Extract location x, y, z
                    match_loc = re.search(r'Location\(x=([-+]?\d*\.\d+|\d+),\s*y=([-+]?\d*\.\d+|\d+),\s*z=([-+]?\d*\.\d+|\d+)\)', line)

                    if match_gnss and match_loc:
                        timestamps.append(float(match_gnss.group(1)))
                        lats.append(float(match_gnss.group(2)))
                        lons.append(float(match_gnss.group(3)))
                        
                        xs.append(float(match_loc.group(1)))
                        ys.append(float(match_loc.group(2)))
                        zs.append(float(match_loc.group(3)))

        except Exception as e:
            print(f"Error reading {gnss_file_path}: {e}")
            continue

        if not lats:
            print(f"No valid GNSS data found for {town_dir_name}")
            continue

        # Calculate speeds
        import numpy as np
        
        ts_arr = np.array(timestamps)
        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        zs_arr = np.array(zs)
        lats_arr = np.array(lats)
        lons_arr = np.array(lons)

        dt = np.diff(ts_arr)
        dx = np.diff(xs_arr)
        dy = np.diff(ys_arr)
        dz = np.diff(zs_arr)
        
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        dt[dt == 0] = 1e-6
        inst_speeds = dist / dt
        
        speeds = np.append(inst_speeds, 0)

        # Plotting
        plt.figure(figsize=(12, 10))
        
        # Add arrows
        arrow_interval = 4  # Reduced interval for dense, line-like arrows
        if len(lons) > arrow_interval:
            u = np.diff(lons_arr)
            v = np.diff(lats_arr)
            
            norm = np.sqrt(u**2 + v**2)
            norm[norm == 0] = 1
            u = u / norm
            v = v / norm

            u = np.append(u, 0)
            v = np.append(v, 0)
            
            # Calculate scale to make arrows shorter but dense (approx 0.5% of the plot span)
            span_x = lons_arr.max() - lons_arr.min()
            span_y = lats_arr.max() - lats_arr.min()
            max_span = max(span_x, span_y) if max(span_x, span_y) > 0 else 1.0
            scale_factor = 1.0 / (max_span * 0.005)

            idx = np.arange(0, len(lons), arrow_interval)
            
            # Use speeds for coloring arrows
            Q = plt.quiver(lons_arr[idx], lats_arr[idx], u[idx], v[idx], speeds[idx],
                       angles='xy', scale_units='xy', scale=scale_factor, 
                       cmap='jet', width=0.006, headwidth=5, pivot='mid', zorder=5)
            
            cbar = plt.colorbar(Q)
            cbar.set_label('Speed (m/s)')
        else:
            # Fallback for short trajectories
            sc = plt.scatter(lons, lats, c=speeds, cmap='jet', s=5, label='Trajectory')
            cbar = plt.colorbar(sc)
            cbar.set_label('Speed (m/s)')

        plt.title(f"GNSS Trajectory & Speed - {town_dir_name}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(True)
        plt.axis('equal')

        output_file = os.path.join(output_dir, f"{town_dir_name}.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    visualize_gnss()
