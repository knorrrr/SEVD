
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate BEV image from LiDAR .bin file")
    parser.add_argument("input_path", help="Path to input .bin file")
    parser.add_argument("output_path", help="Path to output .png file")
    parser.add_argument("--range", type=float, default=50.0, help="Plot range in meters")
    args = parser.parse_args()

    # Load point cloud
    try:
        points = np.fromfile(args.input_path, dtype=np.float32).reshape(-1, 4)
    except FileNotFoundError:
        print(f"Error: File not found {args.input_path}")
        return

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Setup plot
    plt.figure(figsize=(10, 10), facecolor='black')
    
    # Use scatter for color mapping
    # Normalize Z for colormap. Assuming lidar Z range roughly -3m to 3m for cars, or check data range.
    # Let's use min/max of the data for dynamic range or fixed range if preferred.
    # Fixed range is usually better for consistency. Let's try -2.5 to 1.0 (approx car height range relative to sensor or ground)
    # But sensor height is usually 0 if ego-centric.
    # Safe bet: dynamic per frame or fixed reasonable range. Let's use dynamic for now for visibility, or standard coloring.
    # Let's use 'viridis' colormap.
    
    plt.scatter(y, x, c=z, s=0.5, cmap='viridis', marker='.')
    
    plt.xlim(-args.range, args.range)
    plt.ylim(-args.range, args.range)
    plt.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Save
    plt.savefig(args.output_path, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.close()
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
