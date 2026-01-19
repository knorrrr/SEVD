
import pickle
import numpy as np

pkl_path = '/media/ssd/SEVD/carla/out/11_towns_each_12000_ticks_20251230_201113/Town01_Opt_30_12_2025_20_11_38/ego0/gnss_flow.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("Type of data:", type(data))
if isinstance(data, dict):
    # Check keys
    keys = list(data.keys())
    print(f"Sample keys: {keys[:5]}")
    print(f"Key type: {type(keys[0])}")

    # Check consecutive frames
    sorted_keys = sorted([k for k in data.keys() if isinstance(k, (int, str))])
    # Try converting to int for sorting if they represent frames
    try:
        sorted_keys = sorted(keys, key=lambda x: int(x))
    except:
        sorted_keys = sorted(keys)

    print(f"\nNum keys: {len(sorted_keys)}")
    if len(sorted_keys) > 1:
        k1 = sorted_keys[0]
        k2 = sorted_keys[1]
        
        v1 = data[k1]
        v2 = data[k2]
        
        print(f"Frame {k1}: Rotation {v1['rotation']}")
        print(f"Frame {k2}: Rotation {v2['rotation']}")
        
        # Check middle of sequence
        mid = len(sorted_keys)//2
        k3 = sorted_keys[mid]
        k4 = sorted_keys[mid+1]
        print(f"Frame {k3}: Rotation {data[k3]['rotation']}")
        print(f"Frame {k4}: Rotation {data[k4]['rotation']}")
        
        diff = data[k4]['rotation'] - data[k3]['rotation']
        print(f"Diff ({k3}->{k4}): {diff}")
        
        # Check translation magnitude for middle frame
        trans = data[k3]['translation']
        speed = np.linalg.norm(trans) * 10 # m/s (assuming 0.1s interval)
        print(f"Speed at {k3}: {speed:.2f} m/s")
