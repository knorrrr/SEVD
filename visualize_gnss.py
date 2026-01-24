import os
import re
import matplotlib.pyplot as plt
import glob
import sys
import os

# Workaround for local 'carla' directory shadowing the installed carla module
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(f"Script dir: {script_dir}")
# print(f"Initial sys.path: {sys.path}")

if script_dir in sys.path:
    sys.path.remove(script_dir)

# Also remove current directory representation if present
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')

try:
    import carla
    print(f"Imported carla from: {carla.__path__ if hasattr(carla, '__path__') else 'unknown'}")
    print(f"Carla has Client: {hasattr(carla, 'Client')}")
except ImportError:
    # Just in case, try to add it back if it failed for some reason or just fail clearly
    print("Warning: Could not import carla.")
    carla = None

if script_dir not in sys.path:
    sys.path.append(script_dir)

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

    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(100.0)
        print("Connected to CARLA server.")
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        print("Road map will not be visualized.")
        client = None

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
        inst_speeds = (dist / dt) * 3.6 # Convert to km/h
        
        speeds = np.append(inst_speeds, 0)

        # Plotting
        plt.figure(figsize=(14, 14)) # Reduced size for stability

        # --- CARLA Road Map Visualization ---
        if client:
            try:
                # town_dir_name is like "Town01_Opt_...", etc.
                # Extract the base map name (e.g., "Town01")
                map_name = town_dir_name.split('_')[0]
                
                # Get current world to check loaded map
                world = client.get_world()
                
                # Check if current world is already the correct town to save time
                # Note: world.get_map().name typically returns "Carla/Maps/TownXX"
                current_map_name = world.get_map().name.split('/')[-1]
                
                if map_name != current_map_name:
                     print(f"Loading world: {map_name} (derived from {town_dir_name})")
                     try:
                        client.load_world(map_name)
                        world = client.get_world()
                     except RuntimeError:
                        print(f"Failed to load {map_name}, trying with Opt if applicable or checking available maps...")
                        raise

                carla_map = world.get_map()
                
                # Use generate_waypoints for dense coverage to handle curves correctly
                print(f"Generating dense waypoints for {town_dir_name}...")
                waypoints = carla_map.generate_waypoints(1.0) # Every 1 meters
                
                # DEBUG: Analyze lane types
                from collections import Counter
                type_counts = Counter([wp.lane_type for wp in waypoints])
                
                # --- Map Visualization Helpers ---
                def get_lane_color(lane_type):
                    # Colors for different lane types - Grayscale palette
                    if lane_type == carla.LaneType.Driving:
                        return '#A0A0A0' # Medium Gray for driving (Road)
                    # Other types not drawn due to filter, but good to keep palette
                    elif lane_type == carla.LaneType.Sidewalk:
                        return '#E0E0E0' 
                    elif lane_type == carla.LaneType.Shoulder:
                        return '#D0D0D0' 
                    else:
                        return '#F5F5F5' 

                def get_marking_style(marking_type):
                    # Matplotlib line styles
                    if marking_type == carla.LaneMarkingType.Broken:
                        return '--'
                    elif marking_type == carla.LaneMarkingType.Solid:
                        return '-'
                    elif marking_type == carla.LaneMarkingType.SolidSolid:
                        return '-' 
                    elif marking_type == carla.LaneMarkingType.SolidBroken:
                         return '-' 
                    elif marking_type == carla.LaneMarkingType.BrokenSolid:
                         return '--'
                    else:
                        return None 

                def get_marking_color(marking_color):
                    # Light Gray for markings (visible on Medium Gray, but not stark white)
                    return '#F0F0F0' 
                
                def to_geo(loc):
                    gr = carla_map.transform_to_geolocation(loc)
                    return gr.longitude, gr.latitude

                for wp in waypoints:
                    # Common properties
                    width = wp.lane_width
                    lane_type = wp.lane_type
                    
                    # USER REQUEST: "Only draw lanes" (Driving lanes only)
                    if lane_type != carla.LaneType.Driving:
                        continue
                    
                    # Compute corners for a small segment around the waypoint
                    # Since we sample every 2.0m, let's draw a segment of length ~2.0m
                    # Use a slightly larger length (e.g., 2.2m) to overlap and avoid gaps
                    segment_length = 2.2 
                    half_len = segment_length * 0.5
                    half_width = width * 0.5

                    fwd = wp.transform.get_forward_vector()
                    right = carla.Location(x=fwd.y, y=-fwd.x, z=0)
                    
                    center = wp.transform.location
                    
                    # Points: 
                    # p1 = center - fwd*half_len
                    # p2 = center + fwd*half_len
                    # Then offset by width
                    
                    c_back = center - fwd * half_len
                    c_front = center + fwd * half_len
                    
                    p1_l = c_back - right * half_width
                    p1_r = c_back + right * half_width
                    p2_r = c_front + right * half_width
                    p2_l = c_front - right * half_width

                    lon1_l, lat1_l = to_geo(p1_l)
                    lon1_r, lat1_r = to_geo(p1_r)
                    lon2_r, lat2_r = to_geo(p2_r)
                    lon2_l, lat2_l = to_geo(p2_l)
                    
                # ... (Map visualization code)
                # To add "more info", we can ensure we are drawing all lane types clearly.
                # Let's add a border to the lanes to make them distinct.
                
                    # 1. Draw Surface
                    color = get_lane_color(lane_type)
                    # Add Dark Gray border for better definition
                    edge_color = '#606060'
                    plt.fill([lon1_l, lon1_r, lon2_r, lon2_l], 
                             [lat1_l, lat1_r, lat2_r, lat2_l], 
                             color=color, alpha=1.0, zorder=1, edgecolor='none')
                    
                    # Draw only longitudinal edges (parallel to flow) to avoid "chopped" look
                    plt.plot([lon1_l, lon2_l], [lat1_l, lat2_l], color=edge_color, linewidth=0.2, zorder=1)
                    plt.plot([lon1_r, lon2_r], [lat1_r, lat2_r], color=edge_color, linewidth=0.2, zorder=1)
                    
                    # 2. Draw Markings (Restored)
                    # Right Marking
                    r_marking = wp.right_lane_marking
                    if r_marking.type != carla.LaneMarkingType.NONE:
                        style = get_marking_style(r_marking.type)
                        c = get_marking_color(r_marking.color)
                        lw = 0.5
                        if r_marking.type == carla.LaneMarkingType.SolidSolid:
                            lw = 0.8
                        
                        if style:
                            plt.plot([lon1_r, lon2_r], [lat1_r, lat2_r], color=c, linestyle=style, lw=lw, zorder=2)
                            
                    # Left Marking
                    l_marking = wp.left_lane_marking
                    if l_marking.type != carla.LaneMarkingType.NONE:
                        style = get_marking_style(l_marking.type)
                        c = get_marking_color(l_marking.color)
                        lw = 0.5
                        if l_marking.type == carla.LaneMarkingType.SolidSolid:
                            lw = 0.8 
                        
                        if style:
                            plt.plot([lon1_l, lon2_l], [lat1_l, lat2_l], color=c, linestyle=style, lw=lw, zorder=2)

            except Exception as e:
                print(f"Error visualizing map for {town_dir_name}: {e}")
                import traceback
                traceback.print_exc()
        # ------------------------------------
        
        # Plot Trajectory as a continuous colored line
        from matplotlib.collections import LineCollection
        
        points = np.array([lons_arr, lats_arr]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, 100) # Fixed range 0-100 km/h
        lc = LineCollection(segments, cmap='jet', norm=norm, zorder=5)
        
        # Set the values used for colormapping
        lc.set_array(speeds[:-1]) # Use speed at start of segment
        lc.set_linewidth(2.0) # Continuous smooth line
        
        ax = plt.gca()
        ax.add_collection(lc)
        
        # Start and Stop markers
        # Start
        ax.scatter(lons_arr[0], lats_arr[0], c='black', s=60, zorder=10, edgecolors='white', linewidths=1.5)
        ax.annotate('Start', xy=(lons_arr[0], lats_arr[0]), xytext=(-5, 5), textcoords='offset points',
                    color='black', fontsize=12, zorder=11, weight='bold', ha='right', va='bottom')
        
        # Stop
        ax.scatter(lons_arr[-1], lats_arr[-1], c='black', s=60, zorder=10, edgecolors='white', linewidths=1.5)
        ax.annotate('Stop', xy=(lons_arr[-1], lats_arr[-1]), xytext=(5, -5), textcoords='offset points',
                    color='black', fontsize=12, zorder=11, weight='bold', ha='left', va='top')
        
        # Remove title
        # plt.title(f"GNSS Trajectory & Speed - {town_dir_name}")
        
        plt.grid(False)
        
        # Fix aspect ratio for Lat/Lon
        mean_lat = np.mean(lats_arr)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        plt.axis('off') 
        
        # Force a draw so we can get limits? or just use data limits.
        # It's better to trust the autoscale from plotted elements (map + traj).
        # We need to autoscale manually since add_collection doesn't always trigger it correctly for limits with aspect?
        # Actually plt.fill updates limits.
        
        # Add Scale Bar (Position: Bottom Left)
        # Calculate degrees for 50m
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 40075000.0 * np.cos(np.deg2rad(mean_lat)) / 360.0
        
        target_scale_m = 50.0
        scale_len_lon = target_scale_m / meters_per_deg_lon
        
        # Get current limits
        ax.autoscale_view()
        lon_min, lon_max = ax.get_xlim()
        lat_min, lat_max = ax.get_ylim()
        
        # 1. Scale Bar in Axes Coords
        span_lon = lon_max - lon_min
        if span_lon > 1e-9:
             scale_frac = scale_len_lon / span_lon
             
             # Draw line in axes coords (Lowered to 0.01)
             ax.plot([0.05, 0.05 + scale_frac], [0.01, 0.01], transform=ax.transAxes, color='black', linewidth=2, zorder=10)
             ax.text(0.05 + scale_frac/2, 0.015, f'{int(target_scale_m)}m', transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='black', zorder=10)
             
             # 2. Colorbar next to it
             # Gap of 0.05
             cbar_x_start = 0.05 + scale_frac + 0.05
             cax = ax.inset_axes([cbar_x_start, 0.01, 0.25, 0.012]) # Thin horizontal bar
             cbar = plt.colorbar(lc, cax=cax, orientation='horizontal')
             cbar.set_label('Speed (km/h)', fontsize=7, labelpad=2) # Labelpad might need adjustment if too low?
             cbar.ax.tick_params(labelsize=6, length=2)
             
        # Determine strict bounds based on map + trajectory
        # This can be tricky because map is huge. Let's stick to auto-scale but maybe center on trajectory?
        # Matplotlib defaults are usually okay if we plot everything.
        # But if the map is huge and trajectory is small, we might want to zoom in on trajectory?

        output_file = os.path.join(output_dir, f"{town_dir_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white') # Making background black might look cool? Or stick to white.
        # Let's use default white background but maybe set facecolor='w' explicitly if 'off' makes it transparent.
        # Actually, let's keep it simple.

        plt.close()
        print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    visualize_gnss()
