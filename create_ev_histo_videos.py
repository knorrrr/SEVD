import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
from tqdm import tqdm

def create_histogram_frame(npz_file):
    """
    1つのNPZファイルから単一の[C, H, W]形式のヒストグラムデータを読み込み、
    可視化した画像(OpenCV形式 BGR)を返す関数。
    """
    # --- 1. ファイルの読み込み ---
    if not os.path.exists(npz_file):
        print(f"警告: ファイルが見つかりません: {npz_file}")
        return None

    try:
        data = np.load(npz_file)
        if 'histograms' not in data:
            print(f"警告: '{npz_file}' に 'histograms' データが見つかりません。スキップします。")
            return None
        # 3次元の単一ヒストグラムデータを読み込む
        hist = data['histograms']
        if hist.ndim != 3:
            print(f"警告: '{npz_file}' のデータは3次元ではありません (形状: {hist.shape})。スキップします。")
            return None
    except Exception as e:
        print(f"エラー: '{npz_file}' の読み込み中に問題が発生しました: {e}")
        return None

    num_channels, height, width = hist.shape
    num_cols = 1 + num_channels  # (合成画像 + 各チャンネル画像)

    # --- OpenCVによる画像生成 (高速化) ---
    
    # 1. 各画像の準備
    # カラー合成画像
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    if num_channels == 1:
        gray_ch = (hist[0] > 0) * 255
        color_image[:, :, 0] = gray_ch
        color_image[:, :, 1] = gray_ch
        color_image[:, :, 2] = gray_ch
    elif num_channels == 2:
        color_image[:, :, 0] = (hist[0] > 0) * 255  # Red
        color_image[:, :, 2] = (hist[1] > 0) * 255  # Blue
    else:
        color_image[:, :, 0] = (hist[0] > 0) * 255  # Red
        color_image[:, :, 1] = (hist[1] > 0) * 255  # Green
        if num_channels >= 3:
            color_image[:, :, 2] = (hist[2] > 0) * 255  # Blue

    # 各チャンネル画像 (グレースケール -> BGR変換)
    channel_images = []
    for i in range(num_channels):
        # 正規化 (0-255)
        max_val = max(1, hist[i].max())
        norm_hist = (hist[i] / max_val * 255).astype(np.uint8)
        ch_img_bgr = cv2.cvtColor(norm_hist, cv2.COLOR_GRAY2BGR)
        channel_images.append(ch_img_bgr)

    # 2. レイアウトの計算
    # 余白やテキスト領域の設定
    padding = 10
    text_height = 30
    title_height = 40
    
    # 全体の幅と高さ
    total_width = (width * num_cols) + (padding * (num_cols + 1))
    total_height = height + padding * 2 + text_height + title_height
    
    # キャンバス作成 (白背景)
    canvas = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
    
    # 3. 画像の配置とテキスト描画
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)
    
    # 全体タイトル
    filename = os.path.basename(npz_file)
    cv2.putText(canvas, f"File: {filename}", (padding, 25), font, 0.7, text_color, 2)
    
    current_x = padding
    y_offset = title_height + padding
    
    # Combined Image
    canvas[y_offset:y_offset+height, current_x:current_x+width] = color_image
    cv2.putText(canvas, "Combined Image", (current_x, y_offset - 5), font, font_scale, text_color, font_thickness)
    current_x += width + padding
    
    # Channel Images
    for i, ch_img in enumerate(channel_images):
        canvas[y_offset:y_offset+height, current_x:current_x+width] = ch_img
        cv2.putText(canvas, f"Channel {i}", (current_x, y_offset - 5), font, font_scale, text_color, font_thickness)
        current_x += width + padding
        
    return canvas


import concurrent.futures

def create_histogram_video(town_dir, output_base_dir):
    """
    指定されたTownディレクトリ内のヒストグラムデータを動画に変換する
    """
    town_name = os.path.basename(town_dir)
    # dvs_camera-hist-frontフォルダを探す
    # ego0/dvs_camera-hist-front を想定
    input_dir = os.path.join(town_dir, "ego0", "dvs_camera-hist-front")
    
    if not os.path.exists(input_dir):
        # ego0がない場合やパスが違う場合のフォールバック検索
        found_dirs = glob.glob(os.path.join(town_dir, "**", "dvs_camera-hist-front"), recursive=True)
        if found_dirs:
            input_dir = found_dirs[0]
        else:
            print(f"Skipping {town_name}: 'dvs_camera-hist-front' directory not found.")
            return

    npz_files = sorted(glob.glob(os.path.join(input_dir, '*.npz')))
    if not npz_files:
        print(f"Skipping {town_name}: No .npz files found in {input_dir}")
        return

    output_video_path = os.path.join(output_base_dir, f"{town_name}_histogram.mp4")
    print(f"Processing {town_name}: {len(npz_files)} files -> {output_video_path}")

    fps = 10
    out = None

    # tqdmで進捗を表示 (position引数を使って重ならないようにする工夫もできるが、
    # 並列数が多いと乱れるため、ここではシンプルに leave=False で表示)
    for npz_file in tqdm(npz_files, desc=f"{town_name}", leave=False):
        frame = create_histogram_frame(npz_file)
        if frame is None:
            continue
        
        # サイズを縮小して容量削減 (50%)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        if out is None:
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        out.write(frame)

    if out:
        out.release()
        print(f"Done: {output_video_path}")
    else:
        print(f"Failed to create video for {town_name}")

def main():
    parser = argparse.ArgumentParser(description="ルートディレクトリ下の各Townフォルダ内のNPZファイルを可視化し、動画として保存します。")
    parser.add_argument("input_root", help="Townフォルダを含むルートディレクトリ (例: .../11_towns_each_6000_ticks...)")
    parser.add_argument("--output", "-o", help="動画の出力先ディレクトリ (デフォルト: input_root/videos)", default=None)
    
    args = parser.parse_args()
    
    input_root = args.input_root
    base_output_dir = args.output if args.output else input_root
    output_dir = os.path.join(base_output_dir, "ev_videos")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Scanning directory: {input_root}")
    
    # Townごとのディレクトリを検索
    town_dirs = [d for d in glob.glob(os.path.join(input_root, "Town*")) if os.path.isdir(d)]
    
    if not town_dirs:
        print("No Town directories found directly under the input path.")
        return

    print(f"Found {len(town_dirs)} Town directories.")
    
    # 並列処理の設定
    max_workers = min(len(town_dirs), os.cpu_count() or 4)
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(create_histogram_video, town_dir, output_dir) for town_dir in town_dirs]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Total Progress"):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()