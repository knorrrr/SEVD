import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from tqdm import tqdm

def process_file(input_file: Path, output_file: Path, voxel_size: float):
    """
    単一の.binファイルを読み込み、ボクセルダウンサンプリングを行い、結果を保存する。

    Args:
        input_file (Path): 入力.binファイルのパス。
        output_file (Path): 出力.binファイルのパス。
        voxel_size (float): ボクセルグリッドのサイズ。
    """
    # print(f"\n--- 処理中: {input_file.name} ---")

    # 1. .binファイルをNumPy配列として読み込む
    # 各点は (x, y, z, intensity) の float32 * 4
    try:
        p_cloud_numpy = np.fromfile(str(input_file), dtype=np.float32).reshape(-1, 4)
        # print(f"元の点群数: {len(p_cloud_numpy)}")
    except Exception as e:
        print(f"エラー: ファイルを読み込めませんでした。{e}")
        return

    # 2. NumPy配列をOpen3DのPointCloudオブジェクトに変換
    # ボクセル化にはXYZ座標のみを使用
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(p_cloud_numpy[:, :3])

    # 3. ボクセルグリッドフィルターを適用してダウンサンプリング
    downsampled_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
    # print(f"ダウンサンプリング後の点群数: {len(downsampled_pcd.points)}")

    # 4. 結果をNumPy配列に戻し、intensity情報を追加
    downsampled_xyz = np.asarray(downsampled_pcd.points)
    
    # 元のintensityは平均化が難しいため、ここではプレースホルダーとして1.0を設定
    # 必要に応じて、最近傍点のintensityを割り当てるなどの処理も可能です
    placeholder_intensity = np.ones((len(downsampled_xyz), 1), dtype=np.float32)
    
    # (x, y, z) と intensity を結合して (N, 4) の配列に戻す
    downsampled_numpy = np.hstack((downsampled_xyz, placeholder_intensity))

    # 5. 新しい.binファイルとして保存
    try:
        downsampled_numpy.astype(np.float32).tofile(str(output_file))
        # print(f"保存しました: {output_file}")
    except Exception as e:
        print(f"エラー: ファイルを保存できませんでした。{e}")


def main():
    """
    メイン関数: コマンドライン引数を処理し、ファイル/フォルダの処理を実行する。
    """
    parser = argparse.ArgumentParser(
        description="指定された.binファイルまたはフォルダ内の点群をボクセルグリッドでダウンサンプリングします。"
    )
    parser.add_argument(
        "paths", type=str, nargs='+', help="入力の.binファイルまたは.binファイルが含まれるフォルダのパス（複数指定可）。"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.1, help="ボクセルグリッドのサイズ（メートル単位）。デフォルト: 0.1"
    )
    args = parser.parse_args()

    for path_str in args.paths:
        # パスをPathオブジェクトに変換
        input_path = Path(path_str).joinpath("lidar-front_filtered")
        
        if not input_path.exists():
            print(f"エラー: 入力パスが存在しません: {input_path}")
            continue

        # 入力が単一ファイルかフォルダかで処理を分岐
        files_to_process = []
        if input_path.is_file() and input_path.suffix == ".bin":
            files_to_process = [input_path]
            # ファイル単体の場合は、親ディレクトリに _downsampled をつけるか、同じディレクトリに別名で保存するかなど検討が必要だが
            # 既存ロジックに合わせて 親ディレクトリ/stem_downsampled/filename.bin とする
            output_dir = input_path.parent / f"{input_path.stem}_downsampled"
        elif input_path.is_dir():
            files_to_process = list(input_path.glob("*.bin"))
            output_dir = input_path.parent / f"{input_path.stem}_downsampled"
        else:
            print(f"スキップ: 有効な.binファイルまたはフォルダではありません: {input_path}")
            continue
            
        if not files_to_process:
            print(f"警告: 処理対象の.binファイルが見つかりませんでした: {input_path}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output Directory: {output_dir}")

        # ファイルを一つずつ処理
        for file_path in tqdm(files_to_process, desc=f"{input_path.name} を処理中"):
            output_file_path = output_dir / file_path.name
            process_file(file_path, output_file_path, args.voxel_size)

if __name__ == "__main__":
    main()