import os
import argparse
import numpy as np
import pickle

def parse_gnss(gnss_path):
    """
    Parses gnss.txt file.
    Returns a dictionary mapping frame number (int) to dict with 'loc' (np.array) and 'rot' (np.array).
    """
    gnss_data = {}
    try:
        with open(gnss_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                
                # Extract frame
                frame_str = parts[0].split('frame=')[1]
                frame = int(frame_str)
                
                # Extract Location
                x = float(line.split('Location(x=')[1].split(',')[0])
                y = float(line.split('y=')[1].split(',')[0])
                z = float(line.split('z=')[1].split(')')[0])
                
                # Extract Rotation
                pitch = float(line.split('Rotation(pitch=')[1].split(',')[0])
                yaw = float(line.split('yaw=')[1].split(',')[0])
                roll = float(line.split('roll=')[1].split(')')[0])
                
                gnss_data[frame] = {
                    'loc': np.array([x, y, z], dtype=np.float32),
                    'rot': np.array([pitch, yaw, roll], dtype=np.float32)
                }
    except Exception as e:
        print(f"Error parsing GNSS file {gnss_path}: {e}")
        return {}
    return gnss_data

def process_directory(ego_dir):
    """
    Processes a single ego directory to generate GNSS flow data.
    """
    if not os.path.exists(ego_dir):
        print(f"Directory not found: {ego_dir}")
        return

    gnss_path = os.path.join(ego_dir, "gnss", "gnss.txt")
    if not os.path.exists(gnss_path):
        print(f"GNSS file not found: {gnss_path}")
        return

    print(f"Processing GNSS data for: {ego_dir}")
    gnss_data = parse_gnss(gnss_path)
    
    if not gnss_data:
        print("No GNSS data parsed.")
        return

    flow_data = {}
    sorted_frames = sorted(gnss_data.keys())
    
    # Calculate flow for sequential frames
    # matching the logic of preprocess_evpcd.py which pairs i with i+1
    for i in range(len(sorted_frames) - 1):
        curr_frame = sorted_frames[i]
        next_frame = sorted_frames[i+1] # Assuming sequence is continuous or we process available pairs
        
        # If dataset might skip frames, we should strictly check.
        # But for now, we compute flow to the NEXT available frame in GNSS data? 
        # Or should we assume strictly +1? 
        # The lidar files dictate the dataset. 
        # Let's just store the pose for each frame and let the merger decide? 
        # No, "model should only read". So we want T/R in the final dict.
        # So we should compute flow to the immediately following frame if it exists.
        
        if next_frame == curr_frame + 1:
            loc_curr = gnss_data[curr_frame]['loc']
            loc_next = gnss_data[next_frame]['loc']
            translation = loc_next - loc_curr
            
            rot_curr = gnss_data[curr_frame]['rot']
            rot_next = gnss_data[next_frame]['rot']
            rotation = rot_next - rot_curr
            
            flow_data[str(curr_frame)] = { # Use string key to match lidar_token usually
                'translation': translation,
                'rotation': rotation
            }
            
    # Save flow data
    save_path = os.path.join(ego_dir, "gnss_flow.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(flow_data, f)
    print(f"Saved GNSS flow data to {save_path} ({len(flow_data)} entries)")

def main():
    parser = argparse.ArgumentParser(description="Process GNSS data to calculate flow.")
    parser.add_argument('--input-dirs', nargs='+', required=True, help='List of ego directories to process')
    args = parser.parse_args()

    for d in args.input_dirs:
        process_directory(d)

if __name__ == "__main__":
    main()
