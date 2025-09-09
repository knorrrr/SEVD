import os
import pickle
import sys

def generate_pkl(bin_dir):
    if not os.path.exists(bin_dir):
        print(f"❌ Error: '{bin_dir}' does not exist。")
        return

    all_data_list = []

    npz_files = sorted([f for f in os.listdir(os.path.join(bin_dir, "dvs_camera-hist-front")) if f.startswith("hist-dvs-") and f.endswith(".npz")])
    bin_files = sorted([f for f in os.listdir(os.path.join(bin_dir, "lidar-front_filtered")) if f.endswith(".bin")])
    if len(bin_files) < 2:
        print("❌ Error: .binファイルが2つ以上必要です。")
        return

    # usable_len = len(bin_files) - (len(bin_files) % 2)  # 奇数なら1つ減らす

    for i in range(0, len(bin_files) - 1):
        ev_frame = npz_files[i + 1]
        lidar_fname = bin_files[i]
        pred_fname = bin_files[i + 1]
        lidar_token = os.path.splitext(lidar_fname)[0]

        lidar_path = os.path.join(os.path.abspath(bin_dir), "lidar-front_filtered" ,lidar_fname)
        pred_lidar_path = os.path.join(os.path.abspath(bin_dir), "lidar-front_filtered" ,pred_fname)
        ev_path = os.path.join(os.path.abspath(bin_dir),"dvs_camera-hist-front", ev_frame)

        all_data_list.append({
            "lidar_path": lidar_path,
            "pred_lidar_path": pred_lidar_path,
            "ev_path": ev_path, 
            "lidar_token": lidar_token
        })

    train_data = []
    val_data = []
    test_data = []

    # 8:1の比率でシーケンシャルに分割
    for i, entry in enumerate(all_data_list):
        print(f"Processing entry {i }/{len(all_data_list)}: {entry['lidar_token']}")
        if (i % 8) == 7:
            val_data.append(entry)
        elif (i % 8) == 6:
            test_data.append(entry)
        else:
            train_data.append(entry)

    parent_dir = os.path.dirname(bin_dir.rstrip("/"))
    info_dir = os.path.join(parent_dir, "ego0")
    os.makedirs(info_dir, exist_ok=True)

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



# 実行エントリポイント
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_evpcd.py <data_directory>")
        sys.exit(1)

    bin_dir = sys.argv[1]
    generate_pkl(bin_dir)
