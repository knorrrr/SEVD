import pickle
import os
import numpy as np

def inspect_pkl(path):
    print(f"Inspecting {path}")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            print(f"Type: list, Length: {len(data)}")
            if len(data) > 0:
                print(f"Sample[0]: {data[0]}")
        elif isinstance(data, dict):
            print(f"Type: dict, Keys: {list(data.keys())}")
            # Print sample of first key
            first_key = list(data.keys())[0]
            print(f"Sample[{first_key}]: {data[first_key]}")
        else:
            print(f"Type: {type(data)}")
            print(data)
            
    except Exception as e:
        print(f"Error loading {path}: {e}")

base_dir = "/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113"
town_dir = os.path.join(base_dir, "Town01_Opt_30_12_2025_20_11_38", "ego0")
gnss_flow_path = os.path.join(town_dir, "gnss_flow.pkl")
train_info_path = os.path.join(base_dir, "train_info.pkl")

inspect_pkl(gnss_flow_path)
print("-" * 20)
inspect_pkl(train_info_path)
