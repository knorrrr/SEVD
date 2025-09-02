import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def analyze_single_file(file_path):
    """単一の.binファイルを読み込み、点群数を表示する。"""
    if not file_path.lower().endswith('.bin'):
        print(f"エラー: 入力は.binファイルである必要があります: {file_path}")
        return

    if not os.path.exists(file_path):
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return

    try:
        # .binファイルは (x, y, z, intensity) の4つのfloat32で構成されていると仮定
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        num_points = points.shape[0]
        print("--- 解析結果 ---")
        print(f"ファイル: {os.path.basename(file_path)}")
        print(f"点群数: {num_points} 点")
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました {os.path.basename(file_path)} - {e}")

def analyze_directory(directory_path):
    """指定されたディレクトリ内の全.binファイルの点群数を数え、結果をプロットする。"""
    print(f"ディレクトリをスキャン中: {directory_path}\n")
    point_counts = {}
    file_list = sorted([f for f in os.listdir(directory_path) if f.lower().endswith('.bin')])

    if not file_list:
        print("エラー: ディレクトリ内に.binファイルが見つかりませんでした。")
        return

    for filename in file_list:
        file_path = os.path.join(directory_path, filename)
        try:
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
            num_points = points.shape[0]
            point_counts[filename] = num_points
        except Exception as e:
            print(f"警告: ファイルの読み込みに失敗しました {filename} - {e}")
            point_counts[filename] = 0

    print("--- 点群数一覧 ---")
    for filename, count in point_counts.items():
        print(f"{filename}: {count} 点")

    # Matplotlibで結果をプロット
    filenames = list(point_counts.keys())
    counts = list(point_counts.values())
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(filenames))
    bars = ax.barh(y_pos, counts, align='center', color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filenames)
    ax.invert_yaxis()
    ax.set_xlabel('点群数 (Number of Points)')
    ax.set_title('各.binファイルに含まれる点群数')
    ax.bar_label(bars, padding=5)
    fig.tight_layout()
    print("\nグラフウィンドウを閉じてプログラムを終了します。")
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("使い方:")
        print("  単一ファイルの場合: python analyze_points.py <.binファイルのパス>")
        print("  フォルダの場合:   python analyze_points.py <.binファイルが含まれるフォルダのパス>")
        return
    
    target_path = sys.argv[1]

    if os.path.isfile(target_path):
        analyze_single_file(target_path)
    elif os.path.isdir(target_path):
        analyze_directory(target_path)
    else:
        print(f"エラー: 指定されたパスが見つかりません: {target_path}")

if __name__ == "__main__":
    main()