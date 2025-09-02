import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def save_histogram_visualization(npz_file, output_dir):
    """
    1つのNPZファイルから単一の[C, H, W]形式のヒストグラムデータを読み込み、
    1枚のPNG画像として保存する関数。
    """
    # --- 1. ファイルの読み込み ---
    if not os.path.exists(npz_file):
        print(f"警告: ファイルが見つかりません: {npz_file}")
        return

    try:
        data = np.load(npz_file)
        if 'histograms' not in data:
            print(f"警告: '{npz_file}' に 'histograms' データが見つかりません。スキップします。")
            return
        # 3次元の単一ヒストグラムデータを読み込む
        hist = data['histograms']
        if hist.ndim != 3:
            print(f"警告: '{npz_file}' のデータは3次元ではありません (形状: {hist.shape})。スキップします。")
            return
    except Exception as e:
        print(f"エラー: '{npz_file}' の読み込み中に問題が発生しました: {e}")
        return

    num_channels, height, width = hist.shape
    num_cols = 1 + num_channels  # (合成画像 + 各チャンネル画像)

    print(f"処理中... ファイル: {os.path.basename(npz_file)}, 形状: {hist.shape}")

    # --- 2. Matplotlibによるプロット準備 ---
    # 1行だけのプロットを作成
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5.5))

    # プロットが1つだけの場合でもaxesがリストになるように調整
    if num_cols == 1:
        axes = [axes]

    # --- データの準備 ---
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # チャンネル数に応じてRGBにマッピング
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

    # --- プロットの描画 ---
    # カラー合成画像を表示
    axes[0].imshow(color_image)
    axes[0].set_title("Combined Image")
    axes[0].axis('off')

    # 各チャンネルをグレースケールで表示
    for i in range(num_channels):
        ax_channel = axes[i + 1]
        ax_channel.imshow(hist[i], cmap='gray', vmin=0, vmax=max(1, hist[i].max()))
        ax_channel.set_title(f"Channel {i}")
        ax_channel.axis('off')

    fig.suptitle(f"File: {os.path.basename(npz_file)}", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 3. 画像をファイルに保存 ---
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    output_filename = f"{base_name}.png"
    output_path = os.path.join(output_dir, output_filename)

    plt.savefig(output_path)
    plt.close(fig)  # メモリ解放のため、プロットを閉じる

    print(f"=> 保存しました: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ディレクトリ内の各NPZファイル([C,H,W]形式)をPNGにまとめて可視化し、保存します。")
    parser.add_argument("--dir", help="dataset ディレクトリ")
    args = parser.parse_args()

    input_dir = os.path.join(args.dir, "dvs_camera-hist-front")
    output_dir = os.path.join(args.dir, "dvs_camera-hist-img-front")
    os.makedirs(output_dir, exist_ok=True)

    npz_files = glob.glob(os.path.join(input_dir, '*.npz'))

    if not npz_files:
        print(f"エラー: '{input_dir}' 内に.npzファイルが見つかりません。")
        exit()

    print(f"{len(npz_files)} 個の .npz ファイルを処理します...")

    for npz_file in npz_files:
        save_histogram_visualization(npz_file, output_dir)

    print("全ての処理が完了しました。")