import numpy as np
import open3d as o3d
import sys

def main():
    # .binと.plyファイルのパスをコマンドライン引数から受け取る
    if len(sys.argv) < 3:
        print("使い方: python visualize_separate.py <binファイルのパス> <plyファイルのパス>")
        return

    bin_file_path = sys.argv[1]
    ply_file_path = sys.argv[2]

    # --- 1. .bin ファイルの読み込み ---
    try:
        points_bin = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
        print(f"{bin_file_path} から {len(points_bin)} 点を読み込みました。")
        pcd_bin = o3d.geometry.PointCloud()
        
        xyz_points_bin = points_bin[:, :3]
        contiguous_xyz_bin = np.ascontiguousarray(xyz_points_bin)
        pcd_bin.points = o3d.utility.Vector3dVector(contiguous_xyz_bin)
        
        pcd_bin.paint_uniform_color([1.0, 0, 0])  # Red

    except Exception as e:
        print(f"エラー: .binファイルの読み込みに失敗しました。 - {e}")
        return

    # --- 2. .ply ファイルの読み込み ---
    try:
        pcd_ply = o3d.io.read_point_cloud(ply_file_path)
        if not pcd_ply.has_points():
            print(f"エラー: {ply_file_path} には点群データが含まれていません。")
            return
            
        print(f"{ply_file_path} から {len(pcd_ply.points)} 点を読み込みました。")
        pcd_ply.paint_uniform_color([0, 0.651, 0.929])  # Blue

    except Exception as e:
        print(f"エラー: .plyファイルの読み込みに失敗しました。 - {e}")
        return

    # --- 3. 2つのウィンドウを個別に作成して表示 ---
    print("各ウィンドウを閉じてプログラムを終了します。")

    # Visualizerオブジェクトを作成
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='BIN Point Cloud', width=960, height=540, left=50, top=50)
    vis1.add_geometry(pcd_bin)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='PLY Point Cloud', width=960, height=540, left=1020, top=50)
    vis2.add_geometry(pcd_ply)

    # 両方のウィンドウが表示され続けるようにループ処理
    while True:
        # 各ウィンドウのイベントを処理し、更新
        # ウィンドウが閉じられたらループを抜ける
        if not vis1.poll_events() or not vis2.poll_events():
            break
        vis1.update_renderer()
        vis2.update_renderer()

    # ウィンドウを破棄
    vis1.destroy_window()
    vis2.destroy_window()


if __name__ == "__main__":
    main()