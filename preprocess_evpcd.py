import os
import pickle
import sys
import argparse

import json

def generate_pkl(bin_dirs, duration):
    all_data_list = []
    summary = {
        "duration_ticks": duration,
        "total_entries": 0,
        "maps_details": []
    }

    for bin_dir in bin_dirs:
        if not os.path.exists(bin_dir):
            print(f"⚠️ Warning: '{bin_dir}' does not exist. Skipping.")
            continue
        
        print(f"Processing directory: {bin_dir}")

        npz_files = sorted([f for f in os.listdir(os.path.join(bin_dir, "dvs_camera-hist-front")) if f.startswith("hist-dvs-") and f.endswith(".npz")])
        bin_files = sorted([f for f in os.listdir(os.path.join(bin_dir, "lidar-front_filtered_downsampled")) if f.endswith(".bin")])
        
        if len(bin_files) < 2:
            print(f"⚠️ Warning: Not enough .bin files in {bin_dir}. Skipping.")
            continue

        # usable_len = len(bin_files) - (len(bin_files) % 2)  # 奇数なら1つ減らす
        
        entries_in_this_dir = 0
        for i in range(0, len(bin_files) - 1):
            ev_frame = npz_files[i + 1]
            lidar_fname = bin_files[i]
            pred_fname = bin_files[i + 1]
            lidar_token = os.path.splitext(lidar_fname)[0]

            lidar_path = os.path.join(os.path.abspath(bin_dir), "lidar-front_filtered_downsampled" ,lidar_fname)
            pred_lidar_path = os.path.join(os.path.abspath(bin_dir), "lidar-front_filtered_downsampled" ,pred_fname)
            ev_path = os.path.join(os.path.abspath(bin_dir),"dvs_camera-hist-front", ev_frame)

            all_data_list.append({
                "lidar_path": lidar_path,
                "pred_lidar_path": pred_lidar_path,
                "ev_path": ev_path, 
                "lidar_token": lidar_token
            })
            entries_in_this_dir += 1

        # Summary info for this map
        run_dir_name = os.path.basename(os.path.dirname(bin_dir))
        summary["maps_details"].append({
            "map_run_id": run_dir_name,
            "directory": bin_dir,
            "entries_count": entries_in_this_dir
        })
        summary["total_entries"] += entries_in_this_dir

    if not all_data_list:
        print("❌ Error: No valid data found in any directory.")
        return

    train_data = []
    val_data = []
    test_data = []

    # 8:1の比率でシーケンシャルに分割
    for i, entry in enumerate(all_data_list):
        # print(f"Processing entry {i }/{len(all_data_list)}: {entry['lidar_token']}")
        if (i % 8) == 7:
            val_data.append(entry)
        elif (i % 8) == 6:
            test_data.append(entry)
        else:
            train_data.append(entry)

    # 保存先は最初のディレクトリの親ディレクトリ (ego0の親、つまり実行IDディレクトリ) に保存するか、
    # あるいは共通の場所が良いが、ここでは最初の入力ディレクトリの親(ego0)に保存する形を維持しつつ、
    # 複数マップの場合はどうするか... 
    # ユーザー要望は「出力となるpklファイルは1つ」なので、最初のディレクトリの親にまとめて保存します。
    # ただし、collect_data.shの構造上、各マップは別々の実行IDディレクトリになる可能性があるが、
    # 今回の改修で1回のcollect_data.sh実行で複数マップ回すなら、
    # 出力先構造: out/MapName_Date/ego0/...
    # これらをまとめるなら、out/直下か、あるいはスクリプト実行時のカレントディレクトリなどが無難だが、
    # 既存の流儀に従い、最初の入力ディレクトリの親(ego0)に保存する。
    
    first_bin_dir = bin_dirs[0]
    # first_bin_dir is like .../x_towns_eachxxx/TownXX.../ego0
    # We want to save pkls in .../x_towns_eachxxx
    town_dir = os.path.dirname(first_bin_dir.rstrip("/"))
    info_dir = os.path.dirname(town_dir)
    
    os.makedirs(info_dir, exist_ok=True)

    print(f"Saving combined pkl files to: {info_dir}")

    # train_info.pklを保存
    train_save_path = os.path.join(info_dir, "train_info.pkl")
    with open(train_save_path, "wb") as f:
        pickle.dump(train_data, f)
    print(f"✅ 学習用データ {len(train_data)} エントリを {train_save_path} に保存しました。")

    # val_info.pklを保存
    val_save_path = os.path.join(info_dir, "val_info.pkl")
    with open(val_save_path, "wb") as f:
        pickle.dump(val_data, f)
    print(f"✅ 検証用データ {len(val_data)} エントリを {val_save_path} に保存しました。")

    test_save_path = os.path.join(info_dir, "test_info.pkl")
    with open(test_save_path, "wb") as f:
        pickle.dump(test_data, f)
    print(f"✅ テスト用データ {len(test_data)} エントリを {test_save_path} に保存しました。")

    # データセット概要(JSON)を保存
    summary["split_counts"] = {
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data)
    }
    summary_save_path = os.path.join(info_dir, "dataset_summary.json")
    with open(summary_save_path, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"✅ データセット概要を {summary_save_path} に保存しました。")


# 実行エントリポイント
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge data from multiple directories into single pkl files.")
    parser.add_argument('data_directories', nargs='+', help='List of data directories to process')
    parser.add_argument('--duration', type=int, default=None, help='Duration in ticks used for data collection')
    
    args = parser.parse_args()

    generate_pkl(args.data_directories, args.duration)
