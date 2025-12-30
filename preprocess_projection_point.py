import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def build_projection_matrix(w, h, fov):
    """
    指定された幅, 高さ, 画角(FOV)からカメラの内部パラメータ行列(K)を計算する。
    (元のコードから流用)
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def filter_lidar_by_projection(lidar_path, image_path, output_bin_path, lidar_to_camera_matrix, k_matrix):
    """
    LiDAR点群をカメラに投影し、投影できた点だけをフィルタリングして.binファイルに保存する。
    """
    # 1. カメラ画像の読み込み (幅と高さを取得するためだけ)
    try:
        image = Image.open(image_path)
        image_w, image_h = image.size
    except FileNotFoundError:
        print(f"[エラー] 画像ファイルが見つかりません: {image_path}")
        return

    # 2. LiDAR点群(.bin)の読み込み
    try:
        p_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    except FileNotFoundError:
        print(f"[エラー] LiDARファイルが見つかりません: {lidar_path}")
        return

    # 3. 座標変換と投影 (元のコードのロジックをそのまま利用)
    points_3d = p_cloud[:, :3]
    local_lidar_points = np.r_[points_3d.T, [np.ones(len(points_3d))]]
    camera_points_ue4 = np.dot(lidar_to_camera_matrix, local_lidar_points)
    point_in_camera_coords = np.array([
        camera_points_ue4[1],
        camera_points_ue4[2] * -1,
        camera_points_ue4[0]
    ])
    points_2d = np.dot(k_matrix, point_in_camera_coords)
    # Z=0の除算エラーを避けるため、非常に小さい値を追加
    points_2d[2, points_2d[2] == 0] = 1e-6
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]
    ])

    # 4. フィルタリングマスクの作成 (ここが核心部分)
    points_2d = points_2d.T
    # 画像範囲内(0 < x < width, 0 < y < height)かつカメラ前方(z > 0)の点を選択
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    
    # 5. 元の点群をマスクでフィルタリングし、新しい.binファイルとして保存
    filtered_p_cloud = p_cloud[points_in_canvas_mask]
    
    # 結果を.binファイルとして保存
    filtered_p_cloud.astype(np.float32).tofile(str(output_bin_path))

    # print(f"-> 元の点群数: {len(p_cld)} -> フィルタ後: {len(filtered_p_cloud)}")
    # print(f"   保存しました: {output_bin_path}")


def main(args):
    """
    メイン関数: ファイルを処理し、フィルタリングを実行する
    """
    # --- ▼▼▼ 要設定項目 ▼▼▼ ---
    
    # 1. カメラの特性 (Camera Intrinsics)
    CAMERA_FOV = 60.4

    # 2. LiDARからカメラへの相対的な位置・姿勢 (Lidar to Camera Extrinsics)
    from scipy.spatial.transform import Rotation as R
    relative_translation = np.array([0.0, 0.0, 0.0]) 
    relative_rotation_deg = np.array([0.0, 0.0, 0.0]) 
    
    rotation_matrix = R.from_euler('xyz', relative_rotation_deg, degrees=True).as_matrix()
    lidar_to_camera_matrix = np.identity(4)
    lidar_to_camera_matrix[:3, :3] = rotation_matrix
    lidar_to_camera_matrix[:3, 3] = relative_translation
    lidar_to_camera_matrix = np.linalg.inv(lidar_to_camera_matrix)

    # --- ▲▲▲ 設定はここまで ▲▲▲ ---

    # Pathオブジェクトを使用してパスを処理
    input_dir = Path(args.input_dir)
    lidar_dir = input_dir / "lidar-front"  # LiDARファイルがあるフォルダ
    image_dir = input_dir / "rgb_camera-front" # 画像ファイルがあるフォルダ
    output_dir = input_dir / "lidar-front_filtered"

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 入力ディレクトリ内の.binファイルを検索
    lidar_files = sorted(list(lidar_dir.glob("*.bin")))
    if not lidar_files:
        print(f"[エラー] LiDARディレクトリに.binファイルが見つかりません: {lidar_dir}")
        return

    # 最初の画像からサイズを取得し、K行列を計算
    first_image_path = image_dir / (lidar_files[0].stem + '.png')
    try:
        with Image.open(first_image_path) as img:
            image_w, image_h = img.size
        k_matrix = build_projection_matrix(image_w, image_h, CAMERA_FOV)
        print(f"画像サイズ: {image_w}x{image_h}, カメラFOV: {CAMERA_FOV}度 でK行列を計算しました。")
    except FileNotFoundError:
        print(f"[エラー] 対応する画像ファイルが見つかりません: {first_image_path}")
        return

    # ファイルをループして処理
    print(f"Process Directory: {lidar_dir}")
    for lidar_path in tqdm(lidar_files, desc="ファイルを処理中"):
        basename = lidar_path.stem
        image_path = image_dir / (basename + '.png')
        output_bin_path = output_dir / (basename + '.bin')
        
        if not image_path.exists():
            print(f"スキップ: 対応画像なし {lidar_path.name}")
            continue

        filter_lidar_by_projection(lidar_path, image_path, output_bin_path, lidar_to_camera_matrix, k_matrix)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='カメラに投影できないLiDAR点群を削除するスクリプト')
    parser.add_argument('--input-dir', type=str, required=True, help='lidar-frontとrgb_camera-frontフォルダが含まれる入力ディレクトリ')
    
    args = parser.parse_args()
    main(args)