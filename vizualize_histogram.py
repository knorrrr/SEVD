import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def save_combined_histogram_visualization(npz_file, output_dir):
    """
    1つのNPZファイルから全てのヒストグラムデータを読み込み、
    それらを縦に並べた1枚のPNG画像として保存する関数。
    """
    # --- 1. ファイルの読み込み ---
    if not os.path.exists(npz_file):
        print(f"警告: ファイルが見つかりません: {npz_file}")
        return

    try:
        data = np.load(npz_file)
        if 'histograms' not in data or len(data['histograms']) == 0:
            print(f"警告: '{npz_file}' に 'histograms' データが見つからないか、空です。スキップします。")
            return
        histograms = data['histograms']
    except Exception as e:
        print(f"エラー: '{npz_file}' の読み込み中に問題が発生しました: {e}")
        return

    num_indices = len(histograms)
    # 最初のヒストグラムの形状を基準にプロットの列数を決定
    num_channels = histograms[0].shape[0]
    num_cols = 1 + num_channels # (合成画像 + 各チャンネル画像)

    print(f"処理中... ファイル: {os.path.basename(npz_file)}, インデックス数: {num_indices}, チャンネル数: {num_channels}")

    # --- 2. Matplotlibによるプロット準備 ---
    # 全インデックスを縦に並べるためのFigureとAxesを作成
    fig, axes = plt.subplots(num_indices, num_cols, figsize=(5 * num_cols, 5.5 * num_indices))
    
    # インデックスが1つの場合でもaxesが2次元配列になるように調整
    if num_indices == 1:
        axes = np.array([axes])

    # NPZファイル内の全てのヒストグラムをループ処理
    for index, hist in enumerate(histograms):
        
        # --- データの準備 ---
        current_num_channels, height, width = hist.shape
        if current_num_channels != num_channels:
            print(f"警告: {npz_file} 内でチャンネル数が一貫していません。最初のチャンネル数({num_channels})に合わせてプロットします。")

        color_image = np.zeros((height, width, 3), dtype=np.uint8)

        # チャンネル数に応じてRGBにマッピング
        if current_num_channels == 1:
            gray_ch = (hist[0] > 0) * 255
            color_image[:, :, 0] = gray_ch
            color_image[:, :, 1] = gray_ch
            color_image[:, :, 2] = gray_ch
        elif current_num_channels == 2:
            color_image[:, :, 0] = (hist[0] > 0) * 255  # Red
            color_image[:, :, 2] = (hist[1] > 0) * 255  # Blue
        else:
            color_image[:, :, 0] = (hist[0] > 0) * 255  # Red
            color_image[:, :, 1] = (hist[1] > 0) * 255  # Green
            if current_num_channels >= 3:
                color_image[:, :, 2] = (hist[2] > 0) * 255  # Blue
        
        # --- プロットの描画 ---
        # カラー合成画像を表示
        ax_combined = axes[index, 0]
        ax_combined.imshow(color_image)
        ax_combined.set_title(f"Combined (Index: {index})")
        ax_combined.axis('off')

        # 各チャンネルをグレースケールで表示
        for i in range(num_channels):
            ax_channel = axes[index, i + 1]
            if i < current_num_channels:
                ax_channel.imshow(hist[i], cmap='gray', vmin=0, vmax=max(1, hist[i].max()))
                ax_channel.set_title(f"Channel {i}")
            else:
                # チャンネル数が足りない場合は空白にする
                ax_channel.set_visible(False)
            ax_channel.axis('off')

    fig.suptitle(f"File: {os.path.basename(npz_file)}", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # --- 3. 画像をファイルに保存 ---
    base_name = os.path.splitext(os.path.basename(npz_file))[0]
    output_filename = f"{base_name}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.savefig(output_path)
    plt.close(fig) # メモリ解放のため、プロットを閉じる

    print(f"=> 保存しました: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ディレクトリ内の各NPZファイルの全ヒストグラムを一枚のPNGにまとめて可視化し、保存します。")
    parser.add_argument("input_dir", help="入力するヒストグラムの.npzファイルが含まれるディレクトリ")
    parser.add_argument("output_dir", help="可視化された画像を保存するディレクトリ")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    npz_files = glob.glob(os.path.join(args.input_dir, '*.npz'))
    
    if not npz_files:
        print(f"エラー: '{args.input_dir}' 内に.npzファイルが見つかりません。")
        exit()
        
    print(f"{len(npz_files)} 個の .npz ファイルを処理します...")
    
    for npz_file in npz_files:
        save_combined_histogram_visualization(npz_file, args.output_dir)

    print("全ての処理が完了しました。")