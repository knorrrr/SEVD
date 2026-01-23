import json
import os
import pickle

def generate_summary(base_dir, suffix):
    subsets = ["stopped", "straight", "curve"]
    
    # Categorization Metadata
    meta = {
        "stopped": {
            "description": "Stationary vehicle frames.",
            "criteria": "speed < 0.01 m/frame (approx 0.1 m/s)"
        },
        "straight": {
            "description": "Moving vehicle frames with low yaw rate.",
            "criteria": "speed >= 0.01 m/frame AND abs(yaw_rate) < 0.2 deg/frame"
        },
        "curve": {
            "description": "Moving vehicle frames with high yaw rate.",
            "criteria": "speed >= 0.01 m/frame AND abs(yaw_rate) >= 0.2 deg/frame"
        }
    }
    
    for subset in subsets:
        dir_name = f"{subset}{suffix}"
        full_path = os.path.join(base_dir, dir_name)
        
        if not os.path.exists(full_path):
            print(f"Directory not found: {full_path}")
            continue
            
        print(f"Generating summary for {subset}...")
        
        summary = {
            "name": dir_name,
            "description": meta[subset]["description"],
            "criteria": meta[subset]["criteria"],
            "splits": {},
            "total_frames": 0
        }
        
        total = 0
        for split in ["train", "val", "test"]:
            pkl_path = os.path.join(full_path, f"{split}_info.pkl")
            if os.path.exists(pkl_path):
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    count = len(data)
                    summary["splits"][split] = count
                    total += count
                except Exception as e:
                    print(f"  Error reading {split}: {e}")
                    summary["splits"][split] = "error"
            else:
                 summary["splits"][split] = 0
        
        summary["total_frames"] = total
        
        # Write JSON
        json_path = os.path.join(full_path, "dataset_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"  Wrote {json_path}")
        print(f"  Total frames: {total}")

if __name__ == "__main__":
    base_path = "/media/ssd/SEVD/carla/out"
    suffix = "_11_towns_each_12000_ticks_20251230_201113"
    generate_summary(base_path, suffix)
