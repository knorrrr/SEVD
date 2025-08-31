#!/usr/bin/env python

import os
import argparse
import numpy as np
from PIL import Image
from matplotlib import cm

# カラーマップの設定 (元コードから流用)
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def build_projection_matrix(w, h, fov):
    """
    指定された幅, 高さ, 画角(FOV)からカメラの内部パラメータ行列(K)を計算する。
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def project_lidar_to_camera(lidar_path, image_path, output_path, lidar_to_camera_matrix, k_matrix, dot_extent=2):
    """
    LiDAR点群を読み込み、相対トランスフォーム行列を使ってカメラ画像に投影する。
    """
    # 1. カメラ画像の読み込み
    try:
        image = Image.open(image_path).convert('RGB')
        im_array = np.array(image)
    except FileNotFoundError:
        print(f"[エラー] 画像ファイルが見つかりません: {image_path}")
        return

    # 2. LiDAR点群(.bin)の読み込み
    try:
        # 各点は (x, y, z, intensity) の float32 * 4
        p_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    except FileNotFoundError:
        print(f"[エラー] LiDARファイルが見つかりません: {lidar_path}")
        return
    
    # 3. 座標変換と投影
    
    # LiDARの点(x,y,z)と強度(intensity)を分離
    points_3d = p_cloud[:, :3]
    intensity = p_cloud[:, 3]

    # 同次座標系に変換 (x, y, z) -> (x, y, z, 1)
    # (点の数, 4) -> (4, 点の数) のために転置
    local_lidar_points = np.r_[points_3d.T, [np.ones(len(points_3d))]]

    # --- ここがメインの変換処理 ---
    # LiDARローカル座標系からカメラローカル座標系(UE4)へ一度に変換
    camera_points_ue4 = np.dot(lidar_to_camera_matrix, local_lidar_points)

    # UE4座標系から標準カメラ座標系へ軸を変換
    # (x, y, z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        camera_points_ue4[1],
        camera_points_ue4[2] * -1,
        camera_points_ue4[0]
    ])

    # K行列を使って3D点群を2D画像平面に投影
    points_2d = np.dot(k_matrix, point_in_camera_coords)

    # 3番目の要素(Z)で正規化
    points_2d = np.array([
        points_2d[0, :] / points_2d[2, :],
        points_2d[1, :] / points_2d[2, :],
        points_2d[2, :]
    ])

    # 4. 描画処理
    
    # 画像範囲外やカメラ後方の点を除外
    image_w, image_h = image.size
    points_2d = points_2d.T
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < image_w) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < image_h) & \
        (points_2d[:, 2] > 0.0)
    
    points_2d = points_2d[points_in_canvas_mask]
    intensity = intensity[points_in_canvas_mask]

    # 描画座標を整数に
    u_coord = points_2d[:, 0].astype(np.int32)
    v_coord = points_2d[:, 1].astype(np.int32)

    # 強度に応じて色を決定
    # 元のスクリプトの調整値を参考
    intensity = 4 * intensity - 3
    color_map = np.array([
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0
    ]).astype(np.uint8).T

    # 5. 点を描画して画像を保存
    dot_extent -= 1 # 描画サイズ調整
    if dot_extent <= 0:
        im_array[v_coord, u_coord] = color_map
    else:
        for i in range(len(points_2d)):
            im_array[
                v_coord[i]-dot_extent : v_coord[i]+dot_extent+1,
                u_coord[i]-dot_extent : u_coord[i]+dot_extent+1
            ] = color_map[i]
            
    # 結果を保存
    output_image = Image.fromarray(im_array)
    output_image.save(output_path)
    print(f"-> 保存しました: {output_path}")


def main(args):
    """
    メイン関数: ファイルを処理し、投影を実行する
    """
    # --- ▼▼▼ 要設定項目 ▼▼▼ ---
    
    # 1. カメラの特性 (Camera Intrinsics)
    #    お使いのカメラの画角(Field of View)を度数で指定してください
    CAMERA_FOV = 60.4

    # 2. LiDARからカメラへの相対的な位置・姿勢 (Lidar to Camera Extrinsics)
    #    LiDARを原点とした時の、カメラの相対的な位置(メートル)と回転(度)を指定
    #    例: LiDARの0.5m前、0.2m上方に、同じ向きでカメラが設置されている場合
    #    位置: x=0.5, y=0.0, z=0.2
    #    回転: roll=0.0, pitch=0.0, yaw=0.0 (LiDARと同じ向き)
    #    ※CARLAの座標系（X:前, Y:右, Z:上）で指定します
    
    from scipy.spatial.transform import Rotation as R

    # 相対位置 (メートル)
    relative_translation = np.array([0.0, 0.0, 0.0]) 
    # 相対回転 (度)
    relative_rotation_deg = np.array([0.0, 0.0, 0.0]) 
    
    # Scipyを使って回転行列を計算
    rotation_matrix = R.from_euler('xyz', relative_rotation_deg, degrees=True).as_matrix()

    # 4x4の相対トランスフォーム行列を作成
    # これが LiDAR座標系 -> カメラ座標系(UE4) への変換を行う
    lidar_to_camera_matrix = np.identity(4)
    lidar_to_camera_matrix[:3, :3] = rotation_matrix
    lidar_to_camera_matrix[:3, 3] = relative_translation

    # 【重要】LiDARから見たカメラの変換なので、逆行列にする
    lidar_to_camera_matrix = np.linalg.inv(lidar_to_camera_matrix)

    # --- ▲▲▲ 設定はここまで ▲▲▲ ---

    # 出力ディレクトリを作成
    if not os.path.exists(os.path.join(args.input_dir, "projection")):
        os.makedirs(os.path.join(args.input_dir, "projection"))

    # 入力ディレクトリ内の.binファイルを検索
    lidar_files = sorted([f for f in os.listdir(os.path.join(args.input_dir, "lidar-front")) if f.endswith('.bin')])
    if not lidar_files:
        print(f"[エラー] 入力ディレクトリに.binファイルが見つかりません: {args.input_dir}")
        return

    # 最初の画像からサイズを取得し、K行列を計算
    first_image_path = os.path.join(args.input_dir,"rgb_camera-front", os.path.splitext(lidar_files[0])[0] + '.png')
    try:
        with Image.open(first_image_path) as img:
            image_w, image_h = img.size
        k_matrix = build_projection_matrix(image_w, image_h, CAMERA_FOV)
        print(f"画像サイズ: {image_w}x{image_h}, カメラFOV: {CAMERA_FOV}度 でK行列を計算しました。")
    except FileNotFoundError:
        print(f"[エラー] 対応する画像ファイルが見つかりません: {first_image_path}")
        print("K行列を計算できません。処理を中断します。")
        return

    # ファイルをループして処理
    for lidar_file in lidar_files:
        basename = os.path.splitext(lidar_file)[0]
        lidar_path = os.path.join(args.input_dir, "lidar-front", lidar_file)
        image_path = os.path.join(args.input_dir, "rgb_camera-front", basename + '.png')
        # output_path = os.path.join(args.output_dir, basename + '.png')
        output_path = os.path.join(args.input_dir, "projection", basename + '.png')
        
        print(f"処理中: {basename}")
        project_lidar_to_camera(lidar_path, image_path, output_path, lidar_to_camera_matrix, k_matrix, args.dot_extent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LiDAR点群をカメラ画像に投影するスクリプト')
    parser.add_argument('--input-dir', type=str, required=True, help='LiDAR(.bin)とカメラ(.png)の入力ファイルが入ったディレクトリ')
    # parser.add_argument('--output-dir', type=str, required=True, help='投影結果の画像(.png)を保存するディレクトリ')
    parser.add_argument('-d', '--dot-extent', type=int, default=2, help='描画する点の大きさ (ピクセル単位)')
    
    args = parser.parse_args()
    main(args)