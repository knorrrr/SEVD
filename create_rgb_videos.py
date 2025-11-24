import os
import argparse
import cv2
import glob
import numpy as np
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# カラーマップの設定
VIRIDIS = np.array(cm.get_cmap('viridis').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_lidar_to_camera_matrix():
    # LiDARからカメラへの相対的な位置・姿勢 (project_lidar_to_image.pyより)
    relative_translation = np.array([0.0, 0.0, 0.0]) 
    relative_rotation_deg = np.array([0.0, 0.0, 0.0]) 
    
    rotation_matrix = R.from_euler('xyz', relative_rotation_deg, degrees=True).as_matrix()
    
    lidar_to_camera_matrix = np.identity(4)
    lidar_to_camera_matrix[:3, :3] = rotation_matrix
    lidar_to_camera_matrix[:3, 3] = relative_translation
    
    # LiDARから見たカメラの変換なので逆行列
    return np.linalg.inv(lidar_to_camera_matrix)

def overlay_lidar_on_image(image, lidar_path, lidar_to_camera_matrix, k_matrix, dot_extent=2):
    """
    OpenCV画像(BGR)にLiDAR点群を投影して描画する
    """
    if not os.path.exists(lidar_path):
        return image

    try:
        # 各点は (x, y, z, intensity) の float32 * 4
        p_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    except Exception as e:
        print(f"Error reading lidar file {lidar_path}: {e}")
        return image

    points_3d = p_cloud[:, :3]
    intensity = p_cloud[:, 3]

    # 同次座標系に変換
    local_lidar_points = np.r_[points_3d.T, [np.ones(len(points_3d))]]

    # 座標変換
    camera_points_ue4 = np.dot(lidar_to_camera_matrix, local_lidar_points)

    # UE4 -> 標準カメラ座標系 (y, -z, x)
    point_in_camera_coords = np.array([
        camera_points_ue4[1],
        camera_points_ue4[2] * -1,
        camera_points_ue4[0]
    ])

    # 投影
    points_2d = np.dot(k_matrix, point_in_camera_coords)

    # 正規化
    # ゼロ除算回避
    with np.errstate(divide='ignore', invalid='ignore'):
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]
        ])

    # 描画範囲フィルタリング
    height, width = image.shape[:2]
    points_2d = points_2d.T
    
    points_in_canvas_mask = \
        (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < width) & \
        (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < height) & \
        (points_2d[:, 2] > 0.0)
    
    points_2d = points_2d[points_in_canvas_mask]
    intensity = intensity[points_in_canvas_mask]

    if len(points_2d) == 0:
        return image

    u_coord = points_2d[:, 0].astype(np.int32)
    v_coord = points_2d[:, 1].astype(np.int32)

    # 色の計算
    intensity = 4 * intensity - 3
    # BGR順にする (OpenCV用)
    color_map = np.array([
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0, # B
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0, # G
        np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0  # R
    ]).astype(np.uint8).T

    # 描画
    # 高速化のため、dot_extentが小さい場合は直接代入、大きい場合は円などを描画
    # ここでは元のロジックに近い矩形描画を行う
    
    # 画像をコピーして描画（元の画像を書き換えない方が安全だが、ここでは上書きする）
    out_img = image.copy()
    
    dot_extent -= 1
    if dot_extent <= 0:
        out_img[v_coord, u_coord] = color_map
    else:
        # ループは遅いが、Pythonのループで実装するか、あるいはマスク処理にするか
        # ここではシンプルにループする
        for i in range(len(points_2d)):
            cv2.circle(out_img, (u_coord[i], v_coord[i]), dot_extent, color_map[i].tolist(), -1)
            
    return out_img

def create_video_from_images(input_dir, output_file, lidar_dir=None, lidar_to_camera_matrix=None, k_matrix=None, fps=10):
    """
    指定されたディレクトリ内の連番画像を動画に変換する
    lidar_dirが指定されている場合はLiDAR点群を投影する
    """
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"Warning: No images found in {input_dir}")
        return

    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    size = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    print(f"Creating video: {output_file} from {len(image_files)} images...")
    
    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            if lidar_dir and lidar_to_camera_matrix is not None and k_matrix is not None:
                # 対応するLiDARファイルを探す
                basename = os.path.splitext(os.path.basename(filename))[0]
                lidar_path = os.path.join(lidar_dir, basename + ".bin")
                img = overlay_lidar_on_image(img, lidar_path, lidar_to_camera_matrix, k_matrix)
            
            out.write(img)
    
    out.release()
    print(f"Done: {output_file}")

def process_town_directory(town_dir, output_base_dir, enable_lidar=True):
    """
    Townごとのディレクトリを処理する
    """
    town_name = os.path.basename(town_dir)
    
    camera_dir = os.path.join(town_dir, "ego0", "rgb_camera-front")
    lidar_dir = os.path.join(town_dir, "ego0", "lidar-front")
    
    if not os.path.exists(camera_dir):
        found_dirs = glob.glob(os.path.join(town_dir, "**", "rgb_camera-front"), recursive=True)
        if found_dirs:
            camera_dir = found_dirs[0]
            # カメラが見つかったら、同じ階層構造でLiDARも探す
            parent_dir = os.path.dirname(camera_dir)
            lidar_dir = os.path.join(parent_dir, "lidar-front")
        else:
            print(f"Skipping {town_name}: 'rgb_camera-front' directory not found.")
            return

    # LiDAR投影の準備
    lidar_to_camera_matrix = None
    k_matrix = None
    target_lidar_dir = None
    
    if enable_lidar:
        if os.path.exists(lidar_dir):
            target_lidar_dir = lidar_dir
            # K行列の計算 (最初の画像からサイズ取得)
            image_files = glob.glob(os.path.join(camera_dir, "*.png"))
            if image_files:
                img = cv2.imread(image_files[0])
                if img is not None:
                    h, w = img.shape[:2]
                    CAMERA_FOV = 60.4 # project_lidar_to_image.pyより
                    k_matrix = build_projection_matrix(w, h, CAMERA_FOV)
                    lidar_to_camera_matrix = get_lidar_to_camera_matrix()
        else:
            print(f"Warning: LiDAR directory not found for {town_name}, skipping projection.")

    output_filename = f"{town_name}.mp4"
    output_path = os.path.join(output_base_dir, output_filename)
    
    create_video_from_images(camera_dir, output_path, target_lidar_dir, lidar_to_camera_matrix, k_matrix)

def main():
    parser = argparse.ArgumentParser(description="Create videos from CARLA rgb_camera-front images.")
    parser.add_argument("input_root", help="Path to the root directory containing Town folders")
    parser.add_argument("--output", "-o", help="Output directory for videos", default=None)
    parser.add_argument("--no-lidar", action="store_true", help="Disable LiDAR projection")
    
    args = parser.parse_args()
    
    input_root = args.input_root
    base_output_dir = args.output if args.output else input_root
    output_dir = os.path.join(base_output_dir, "rgb_videos")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Scanning directory: {input_root}")
    
    town_dirs = [d for d in glob.glob(os.path.join(input_root, "Town*")) if os.path.isdir(d)]
    
    if not town_dirs:
        print("No Town directories found directly under the input path.")
        return

    print(f"Found {len(town_dirs)} Town directories.")
    
    max_workers = min(len(town_dirs), os.cpu_count() or 4)
    print(f"Starting parallel processing with {max_workers} workers...")
    
    import concurrent.futures
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_town_directory, town_dir, output_dir, not args.no_lidar) for town_dir in town_dirs]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Total Progress"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
