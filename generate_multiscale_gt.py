import os
import subprocess
import glob

def main():
    # Hardcoded dataset path
    dataset_base = "/media/ssd/SEVD/carla/out/100Hz_1_towns_each__ticks_20260119_125916"
    
    # Find all Town* ego directories
    ego_dirs = sorted(glob.glob(os.path.join(dataset_base, "Town*")))
    
    # Intervals to generate (in addition to existing 10ms)
    # We want 50ms, 500ms, 1000ms.
    # Base interval is 10ms (1 file).
    # So windows are: 5, 50, 100.
    
    configs = [
        (50, 5),
        (100, 10),
        (500, 50),
        (1000, 100)
    ]
    
    script_ev = "preprocess_ev.py"
    script_pcd = "preprocess_evpcd.py"
    
    for ego_dir in args.ego_dirs:
        print(f"Processing {ego_dir}...")
        
        # 1. Generate Accumulated Histograms
        for ms, window in configs:
            output_dir = os.path.join(ego_dir, f"dvs_hist_accum_{ms}ms")
            # Removed naive skip logic to allow resuming. 
            # preprocess_ev.py now checks individual files.

            print(f"Generating {ms}ms histograms (window={window})...")
            cmd = [
                "python3", script_ev,
                "--dir", ego_dir, # preprocess_ev takes parent of ego?
                                                   # No, --dir /path/to/dataset  
                                                   # It expects structure: dir/dvs_camera-front/*.npz
                                                   # ego_dir IS the dataset dir containing dvs_camera-front.
                "--window_size", str(window),
                "--output_dir", output_dir
            ]
            # preprocess_ev arguments are a bit weird.
            # parser.add_argument('--dir')
            # input_dir = os.path.join(args.dir, "dvs_camera-front")
            # So if we pass ego_dir, it looks for ego_dir/dvs_camera-front. Correct.
            
            subprocess.run(cmd, check=True)
            
    # 2. Generate PKL files
    # We pass ALL ego_dirs to this script
    print("Generating PKL dataset files...")
    cmd_pcd = [
        "python3", script_pcd,
        *ego_dirs,
        "--intervals", "10,50,100,500,1000"
    ]
    subprocess.run(cmd_pcd, check=True)

if __name__ == "__main__":
    main()
