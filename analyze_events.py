import numpy as np
import argparse
import os
import glob

def analyze_npz_file(filepath):
    """
    単一のNPZファイルを分析し、イベント数と時間の幅を出力する関数。
    """
    try:
        # allow_pickle=True は構造化配列の読み込みに必要
        data = np.load(filepath, allow_pickle=True)

        # 'dvs_events' というキーが存在するかチェック
        if 'dvs_events' not in data:
            print(f"警告: '{os.path.basename(filepath)}' に 'dvs_events' キーが見つかりません。スキップします。\n")
            return

        events = data['dvs_events']

        # --- 1. イベントの数を計算 ---
        num_events = len(events)
        if num_events == 0:
            print(f"--- 分析結果: {os.path.basename(filepath)} ---")
            print("  イベント数: 0")
            print("  このファイルにはイベントデータが含まれていません。\n")
            return

        # --- 2. タイムスタンプを抽出 ---
        # 構造化配列のフィールド名 ('t' や 'f2'など) を自動で探す
        fields = events.dtype.fields
        timestamp_key = None
        if 't' in fields:
            timestamp_key = 't'
        elif 'f2' in fields:
            timestamp_key = 'f2'
        else:
            # ユーザー提供のデータ形式 (x, y, t, p) のタプルを想定し、3番目の要素を試す
            try:
                # 構造化配列でない場合のフォールバック
                timestamps = events[:, 2]
            except (IndexError, TypeError):
                 print(f"警告: '{os.path.basename(filepath)}' でタイムスタンプのフィールドが見つかりません。スキップします。\n")
                 return
        
        if timestamp_key:
            timestamps = events[timestamp_key]

        # --- 3. 時間の幅を計算 ---
        min_t = timestamps.min()
        max_t = timestamps.max()
        duration = max_t - min_t

        # タイムスタンプの単位はマイクロ秒(μs)と仮定して、人間に分かりやすい単位に変換
        duration_ms = duration / 1000.0  # ミリ秒
        duration_s = duration / 1000000.0 # 秒

        # --- 4. 結果を出力 ---
        print(f"--- 分析結果: {os.path.basename(filepath)} ---")
        print(f"  イベント総数: {num_events:,} 件")
        print(f"  時間の幅: {duration_s:,.3f} 秒 ({duration_ms:,.3f} ミリ秒)")
        print(f"  (RAW値: {duration:,})")
        print(f"  最小タイムスタンプ: {min_t:,}")
        print(f"  最大タイムスタンプ: {max_t:,}\n")

    except Exception as e:
        print(f"エラー: '{filepath}' の処理中に予期せぬ問題が発生しました: {e}\n")

def main():
    """
    メイン関数。コマンドライン引数を処理し、分析を実行する。
    """
    parser = argparse.ArgumentParser(description="NPZファイル内のDVSイベントデータの数と時間の幅を分析します。")
    parser.add_argument("input_path", help="分析対象の.npzファイル、または.npzファイルが含まれるディレクトリのパス")
    args = parser.parse_args()

    # パスがディレクトリかファイルかを判定
    if os.path.isdir(args.input_path):
        # ディレクトリ内の全ての.npzファイルを検索
        npz_files = sorted(glob.glob(os.path.join(args.input_path, "*.npz")))
        if not npz_files:
            print(f"エラー: ディレクトリ '{args.input_path}' 内に.npzファイルが見つかりません。")
            return
    elif os.path.isfile(args.input_path) and args.input_path.endswith('.npz'):
        npz_files = [args.input_path]
    else:
        print(f"エラー: '{args.input_path}' は有効な.npzファイルまたはディレクトリではありません。")
        return
    
    print(f"合計 {len(npz_files)} 個のファイルを分析します。\n")

    for filepath in npz_files:
        analyze_npz_file(filepath)

if __name__ == '__main__':
    main()