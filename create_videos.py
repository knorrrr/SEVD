import os
import argparse
import cv2
import glob
from tqdm import tqdm

def create_video_from_images(input_dir, output_file, fps=10, quality=25):
    """
    指定されたディレクトリ内の連番画像を動画に変換する
    """
    # 画像ファイルの検索 (png)
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_files:
        print(f"Warning: No images found in {input_dir}")
        return

    # 最初の画像を読み込んでサイズを取得
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Error: Could not read first image {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    size = (width, height)
    
    # 動画作成の準備 (mp4v codec, 軽量化のため画質調整)
    # H.264が使える場合は 'avc1' や 'H264' を試すが、汎用性のため 'mp4v' を使用
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    print(f"Creating video: {output_file} from {len(image_files)} images...")
    
    for filename in image_files:
        img = cv2.imread(filename)
        if img is not None:
            # 画質調整（リサイズはしないが、必要ならここで行う）
            # JPEG圧縮のような画質劣化はVideoWriterのパラメータでは直接指定しにくいが、
            # エンコーディング時に圧縮される。
            out.write(img)
    
    out.release()
    print(f"Done: {output_file}")

def process_town_directory(town_dir, output_base_dir):
    """
    Townごとのディレクトリを処理する
    構造: TownXX_Opt_.../ego0/rgb_camera-front/*.png
    """
    town_name = os.path.basename(town_dir)
    
    # rgb_camera-frontフォルダを探す
    # ego0/rgb_camera-front を想定
    camera_dir = os.path.join(town_dir, "ego0", "rgb_camera-front")
    
    if not os.path.exists(camera_dir):
        # ego0がない場合やパスが違う場合のフォールバック検索
        found_dirs = glob.glob(os.path.join(town_dir, "**", "rgb_camera-front"), recursive=True)
        if found_dirs:
            camera_dir = found_dirs[0]
        else:
            print(f"Skipping {town_name}: 'rgb_camera-front' directory not found.")
            return

    # 出力ファイル名
    output_filename = f"{town_name}.mp4"
    output_path = os.path.join(output_base_dir, output_filename)
    
    create_video_from_images(camera_dir, output_path)

def main():
    parser = argparse.ArgumentParser(description="Create videos from CARLA rgb_camera-front images.")
    parser.add_argument("input_root", help="Path to the root directory containing Town folders (e.g., .../2_towns_each_5_ticks_...)")
    parser.add_argument("--output", "-o", help="Output directory for videos (default: same as input_root)", default=None)
    
    args = parser.parse_args()
    
    input_root = args.input_root
    output_dir = args.output if args.output else input_root
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Scanning directory: {input_root}")
    
    # Townごとのディレクトリを検索
    # パターン: Town* (ディレクトリ)
    town_dirs = [d for d in glob.glob(os.path.join(input_root, "Town*")) if os.path.isdir(d)]
    
    if not town_dirs:
        print("No Town directories found directly under the input path.")
        return

    print(f"Found {len(town_dirs)} Town directories.")
    
    import concurrent.futures
    
    # Use ProcessPoolExecutor for parallel processing
    # Adjust max_workers based on CPU cores, e.g., os.cpu_count() or a fixed number like 4 or 8
    max_workers = min(len(town_dirs), os.cpu_count() or 4)
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_town_directory, town_dir, output_dir) for town_dir in town_dirs]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Total Progress"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
