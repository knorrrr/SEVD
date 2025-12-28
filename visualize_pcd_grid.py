import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Open3D not found. Will attempt simple parsing (only works for simple ASCII/Binary uncompressed).")

def load_pcd(file_path):
    if HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        return points
    else:
        # Fallback for simple pcd extraction if open3d is missing
        points = []
        with open(file_path, 'rb') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore')
                if line.startswith('DATA'):
                    data_type = line.split()[1]
                    if data_type == 'ascii':
                        # Read rest as text
                        data = np.loadtxt(f)
                        points = data[:, :3] # Take XYZ
                    else:
                        print("Binary PCD parsing without Open3D is experimental/not fully implemented in this simple script.")
                        return None
                    break
        return np.array(points)

def load_bin(file_path):
    # Load binary point cloud (x, y, z, intensity)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points[:, :3] # Return XYZ

def main():
    parser = argparse.ArgumentParser(description="Visualize PCD/BIN file with Grid")
    parser.add_argument("input_path", help="Path to input .pcd or .bin file")
    parser.add_argument("--output", default=None, help="Output image path. If not provided, saves as <input_filename>_grid.png")
    parser.add_argument("--range", type=float, default=50.0, help="Plot range X in meters (default: 50.0)")
    parser.add_argument("--range_y", type=float, default=None, help="Plot range Y in meters (default: same as range)")
    parser.add_argument("--grid_size", type=float, default=10.0, help="Grid size in meters (default: 10.0)")
    parser.add_argument("--circle_interval", type=float, default=0.0, help="Interval for concentric circles (default: 0.0, disabled). Uses custom logic if > 0.")
    
    args = parser.parse_args()
    
    range_x = args.range
    range_y = args.range_y if args.range_y is not None else args.range
    
    if not os.path.exists(args.input_path):
        print(f"Error: File {args.input_path} not found.")
        return

    print(f"Loading {args.input_path}...")
    if args.input_path.endswith('.bin'):
        points = load_bin(args.input_path)
    else:
        points = load_pcd(args.input_path)
    
    if points is None or len(points) == 0:
        print("Failed to load points or empty file.")
        return

    print(f"Loaded {len(points)} points.")
    
    # Extract X, Y, Z
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2] # Use Z for color

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Scatter plot
    # Filter points within range for better visualization
    # User requested no negative X
    mask = (x >= 0) & (x < range_x) & (np.abs(y) < range_y)
    x_plot = x[mask]
    y_plot = y[mask]
    z_plot = z[mask]
    
    sc = ax.scatter(x_plot, y_plot, c=z_plot, s=1, cmap='viridis', marker='.')
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    
    # Add concentric circles
    if args.circle_interval > 0:
        radii = []
        # 0 to 20m with 2.5m step (excluding 0)
        curr = 2.5
        while curr <= 20.0 and curr < range_x:
             # Avoid precise float matches issues by checking boundaries loosely
             radii.append(curr)
             curr += 2.5
        
        # > 20m with 10m step
        # Start from 30.0 because 20.0 (or close to it) is covered above
        curr = 30.0
        while curr < range_x:
            radii.append(curr)
            curr += 10.0
            
        for r in radii:
            circle = plt.Circle((0, 0), r, fill=False, color='yellow', linestyle='-', linewidth=0.8, alpha=0.9)
            ax.add_artist(circle)
            
            # Label positioning
            angle = np.pi/8
            # Check Y bounds
            if r * np.sin(angle) > range_y:
                 # Adjust angle to fit in Y range if r > range_y
                 if r > range_y:
                    try:
                        angle = np.arcsin(range_y / r) * 0.9 
                    except ValueError:
                        angle = 0
                 else:
                    angle = 0
            
            if r * np.sin(angle) <= range_y:
               ax.text(r * np.cos(angle), r * np.sin(angle), f'{r:.1f}m', color='yellow', fontsize=6, fontweight='bold', ha='center', va='center')

    # Set ticks based on grid_size
    xticks = np.arange(0, range_x + args.grid_size, args.grid_size)
    yticks = np.arange(-range_y, range_y + args.grid_size, args.grid_size)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    ax.set_xlim(0, range_x)
    ax.set_ylim(-range_y, range_y)
    
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"PCD Visualization: {os.path.basename(args.input_path)}")
    plt.colorbar(sc, label='Z height')
    
    # Make aspect ratio equal
    ax.set_aspect('equal', adjustable='box')

    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(args.input_path))[0]
        output_path = f"{base_name}_grid.png"
    
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
