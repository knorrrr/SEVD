import numpy as np
import open3d as o3d
import sys
import os

def main():
    # コマンドライン引数からファイルのパスを特定
    if len(sys.argv) < 2:
        print("使い方: python visualize_flexible.py <ファイルパス1> [<ファイルパス2>]")
        print("       .bin と .ply ファイルを1つまたは2つ指定できます。")
        return
import numpy as np
import open3d as o3d
import sys
import os

def main():
    # コマンドライン引数からファイルのパスを特定
    if len(sys.argv) < 2:
        print("使い方: python visualize_flexible.py <ファイルパス1> [<ファイルパス2>]")
        print("       .bin と .ply ファイルを1つまたは2つ指定できます。")
        return

    bin_file_path = None
    ply_file_path = None

    # 引数を調べて、拡張子に応じてパスを割り当てる
    for arg in sys.argv[1:]:
        if arg.lower().endswith('.bin'):
            bin_file_path = arg
        elif arg.lower().endswith('.ply'):
            ply_file_path = arg

    if not bin_file_path and not ply_file_path:
        print("エラー: .bin または .ply ファイルが見つかりませんでした。")
        return

    pcd_bin = None
    pcd_ply = None

    # --- 1. .bin ファイルの読み込み (パスが指定されていれば) ---
    if bin_file_path:
        if not os.path.exists(bin_file_path):
            print(f"エラー: .binファイルが見つかりません: {bin_file_path}")
        else:
            try:
                points_bin = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
                print(f"{bin_file_path} から {len(points_bin)} 点を読み込みました。")
                pcd_bin = o3d.geometry.PointCloud()
                pcd_bin.points = o3d.utility.Vector3dVector(points_bin[:, :3])
                pcd_bin.paint_uniform_color([1.0, 0, 0])  # Red
            except Exception as e:
                print(f"エラー: .binファイルの読み込みに失敗しました。 - {e}")

    # --- 2. .ply ファイルの読み込み (パスが指定されていれば) ---
    if ply_file_path:
        if not os.path.exists(ply_file_path):
            print(f"エラー: .plyファイルが見つかりません: {ply_file_path}")
        else:
            try:
                pcd_ply = o3d.io.read_point_cloud(ply_file_path)
                if not pcd_ply.has_points():
                    print(f"警告: {ply_file_path} には点群データが含まれていません。")
                    pcd_ply = None # データがない場合はNoneに戻す
                else:
                    print(f"{ply_file_path} から {len(pcd_ply.points)} 点を読み込みました。")
                    pcd_ply.paint_uniform_color([0, 0.651, 0.929])  # Blue
            except Exception as e:
                print(f"エラー: .plyファイルの読み込みに失敗しました。 - {e}")

    # --- 3. 存在する点群だけを表示 ---
    visualizers = []
    
    if pcd_bin:
        vis_bin = o3d.visualization.Visualizer()
        vis_bin.create_window(window_name='BIN Point Cloud', width=960, height=540, left=50, top=50)
        vis_bin.add_geometry(pcd_bin)
        visualizers.append(vis_bin)

    if pcd_ply:
        vis_ply = o3d.visualization.Visualizer()
        vis_ply.create_window(window_name='PLY Point Cloud', width=960, height=540, left=1020, top=50)
        vis_ply.add_geometry(pcd_ply)
        visualizers.append(vis_ply)

    if not visualizers:
        print("表示できる点群データがありませんでした。プログラムを終了します。")
        return
        
    print("ウィンドウを閉じてプログラムを終了します。")
    while True:
        # アクティブなウィンドウが一つでも残っているかチェック
        active_window_found = False
        for vis in visualizers:
            if vis.poll_events():
                active_window_found = True
            vis.update_renderer()
        
        # 全てのウィンドウが閉じられたらループを抜ける
        if not active_window_found:
            break

    # ウィンドウを破棄
    for vis in visualizers:
        vis.destroy_window()

if __name__ == "__main__":
    main()