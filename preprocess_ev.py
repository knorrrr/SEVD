import argparse
import numpy as np
import torch
import os
import glob

class StackedHistogram:
    """
    イベントデータから、時間と極性を考慮した積層ヒストグラムを生成するクラス。
    """
    def __init__(self, bins, height, width, count_cutoff=None):
        self.bins = bins
        self.height = height
        self.width = width
        self.count_cutoff = count_cutoff

    def get_shape(self):
        return (self.bins, self.height, self.width)

    def get_numpy_dtype(self):
        return np.uint8

    def construct(self, x, y, pol, t, time_bins_per_pol):
        """
        時間と極性に基づいてイベントを異なるチャンネルに割り振る。
        
        Args:
            x (torch.Tensor): イベントのx座標
            y (torch.Tensor): イベントのy座標
            pol (torch.Tensor): イベントの極性 (0 or 1)
            t (torch.Tensor): イベントのタイムスタンプ
            time_bins_per_pol (int): 極性ごとに時間軸を何分割するか
        """
        hist = torch.zeros((self.bins, self.height, self.width), dtype=torch.uint8, device=x.device)

        # ウィンドウ内の時間の範囲を計算
        min_t, max_t = t.min(), t.max()
        time_range = max_t - min_t
        
        # ゼロ除算を避ける
        if time_range == 0:
            time_range = 1

        # タイムスタンプを正規化 (0.0 ~ 1.0) し、時間ビンを計算
        # (t - min_t) / time_range で正規化し、time_bins_per_polを掛ける
        time_bin = ((t - min_t).float() * time_bins_per_pol / time_range).long()
        # time_binが範囲内に収まるようにclamp
        time_bin = torch.clamp(time_bin, 0, time_bins_per_pol - 1)

        # 最終的なチャンネルインデックスを計算
        # 例: time_bins=4の場合
        # pol=0, time_bin=0 -> final_bin=0
        # pol=0, time_bin=3 -> final_bin=3
        # pol=1, time_bin=0 -> final_bin=4
        # pol=1, time_bin=3 -> final_bin=7
        final_bin = pol.long() * time_bins_per_pol + time_bin

        # 座標と極性が有効な範囲内にあるイベントのみを対象とするマスクを作成
        mask = (x >= 0) & (x < self.width) & \
               (y >= 0) & (y < self.height) & \
               (pol >= 0) & (pol < 2) # 極性は0か1のみを想定
        
        x_f, y_f, final_bin_f = x[mask], y[mask], final_bin[mask]

        if x_f.shape[0] == 0:
            return hist

        # (channel, y, x) の3次元インデックスを1次元のフラットなインデックスに変換
        indices = final_bin_f * self.height * self.width + \
                  y_f.long() * self.width + \
                  x_f.long()
        
        counts = torch.bincount(indices, minlength=self.bins * self.height * self.width)
        hist = counts.view(self.bins, self.height, self.width).to(torch.uint8)

        if self.count_cutoff is not None:
            hist = torch.clamp(hist, max=self.count_cutoff)
            
        return hist

# (NPZReader, NPZWriter, downsample_ev_reprは変更なしのため省略)
# ... (前回のコードと同じものをここに挿入) ...
class NPZReader:
    def __init__(self, npz_file, height, width):
        assert os.path.exists(npz_file)
        data = np.load(npz_file, allow_pickle=True, mmap_mode='r')
        events = data['dvs_events']
        self.x = events['f0'] if 'f0' in events.dtype.fields else events['x'] if 'x' in events.dtype.fields else events[:,0]
        self.y = events['f1'] if 'f1' in events.dtype.fields else events['y'] if 'y' in events.dtype.fields else events[:,1]
        self.t = events['f2'] if 'f2' in events.dtype.fields else events['t'] if 't' in events.dtype.fields else events[:,2]
        self.p = events['f3'] if 'f3' in events.dtype.fields else events['pol'] if 'pol' in events.dtype.fields else events[:,3]
        if self.p.dtype == np.bool_: self.p = self.p.astype(np.int64)
        self.height = height
        self.width = width
        self.is_open = True
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.is_open = False
    @property
    def time(self): return self.t
    def get_event_slice(self, idx_start, idx_end):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return dict(
            x=torch.from_numpy(self.x[idx_start:idx_end].astype('int64')).to(device),
            y=torch.from_numpy(self.y[idx_start:idx_end].astype('int64')).to(device),
            p=torch.from_numpy(self.p[idx_start:idx_end].astype('int64')).to(device),
            t=torch.from_numpy(self.t[idx_start:idx_end].astype('int64')).to(device),
        )

class NPZWriter:
    def __init__(self, outfile):
        self.outfile, self.data_list = outfile, []

    def add_data(self, data):
        if isinstance(data, torch.Tensor): data = data.cpu().numpy()
        self.data_list.append(data)

    def close(self):
        if not self.data_list: return
        
        if len(self.data_list) == 1:
            # 要素が1つだけなら、そのまま保存
            final_data = self.data_list[0]
        else:
            # 複数ある場合は従来通りスタックする
            final_data = np.stack(self.data_list, axis=0)
            
        np.savez_compressed(self.outfile, histograms=final_data)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
def downsample_ev_repr(x, scale_factor):
    if x.dim() == 3: x = x.unsqueeze(0)
    x = torch.nn.functional.interpolate(x.float(), scale_factor=scale_factor, mode='nearest')
    return x.squeeze(0).byte()


def process_file(input_npz, output_npz, args, height, width):
    with NPZReader(input_npz, height, width) as reader:
        # ダウンサンプリングする場合、高さと幅を更新
        if args.downsample:
            height //= 2
            width //= 2
            
        hist_generator = StackedHistogram(bins=2 * args.time_bins, height=height, width=width) # count_cutoffは後で適用
        
        with NPZWriter(output_npz) as writer:
            ev_ts = reader.time
            total_events = len(ev_ts)
            window = args.window_events
            
            # --- 変更点 1: 統合ヒストグラム用の入れ物を作成 ---
            # カウントの合計値が255を超える可能性があるので、大きなデータ型(long)で初期化
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            combined_hist = torch.zeros((2 * args.time_bins, height, width), dtype=torch.long, device=device)
            # -----------------------------------------------

            # print(f"合計 {total_events} イベントを処理します...")
            for idx_start in range(0, total_events, window):
                idx_end = min(idx_start + window, total_events)
                if idx_start >= idx_end: continue
                
                ev_window = reader.get_event_slice(idx_start, idx_end)
                ev_repr = hist_generator.construct(ev_window['x'], ev_window['y'], ev_window['p'], ev_window['t'], args.time_bins)
                
                if args.downsample:
                    ev_repr = downsample_ev_repr(ev_repr, scale_factor=0.5)
                
                # --- 変更点 2: writerに都度追加するのではなく、足し合わせる ---
                # ev_repr(uint8)をlongに変換してから加算
                combined_hist += ev_repr.long()
                # -------------------------------------------------------

                print(f"\r進捗: {idx_end}/{total_events} ({idx_end/total_events:.1%})", end="")

            # --- 変更点 3: ループ終了後、最終処理をしてから一度だけwriterに渡す ---
            # count_cutoffをここで適用
            if args.count_cutoff is not None:
                combined_hist = torch.clamp(combined_hist, max=args.count_cutoff)
            
            # 最終的な型をuint8に戻す (必要に応じて変更可能)
            # 注意: 255を超える値は255にクリップされます
            final_hist = combined_hist.to(torch.uint8)
            
            writer.add_data(final_hist)
            # -------------------------------------------------------------------

    print(f"\n処理が完了しました。 {output_npz} に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="イベントデータを時間と極性を考慮した積層ヒストグラムに変換します。")
    parser.add_argument('--dir', help='入力Dataset ディレクトリ（例: /path/to/dataset）')
    parser.add_argument('--height', type=int, default=480, help='イベントデータの高さ（ピクセル数）')
    parser.add_argument('--width', type=int, default=640, help='イベントデータの幅（ピクセル数）')
    parser.add_argument('--time_bins', type=int, default=10, help='時間軸の分割数。最終的なチャンネル数は 2 * time_bins となります。')
    parser.add_argument('--count_cutoff', type=int, default=None, help='ヒストグラムの各ピクセルの最大カウント値。')
    parser.add_argument('--downsample', action='store_true', help='有効にすると、生成されたヒストグラムを2分の1にダウンサンプリングします。')
    parser.add_argument('--window_events', type=int, default=20000, help='1つのヒストグラムを生成するために使用するイベント数。')
    args = parser.parse_args()

    output_dir = os.path.join(args.dir, "dvs_camera-hist-front")
    input_dir = os.path.join(args.dir, "dvs_camera-front")
    os.makedirs(output_dir, exist_ok=True)
    npz_files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    print(f"{len(npz_files)} 個のファイルを処理します。")

    import concurrent.futures
    def process_wrapper(input_npz):
        base = os.path.basename(input_npz)
        output_npz = os.path.join(output_dir, f"hist-{base}")
        print(f"\n[Thread] 入力ファイル: {input_npz} 高さ={args.height}, 幅={args.width}, 合計チャンネル数={2 * args.time_bins}")
        process_file(input_npz, output_npz, args, args.height, args.width)

    max_workers = min(50, len(npz_files)) # 並列数は最大50、またはファイル数まで
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_wrapper, input_npz) for input_npz in npz_files]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"エラー: {e}")

if __name__ == '__main__':
    main()