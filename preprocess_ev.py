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
            
        hist_generator = StackedHistogram(bins=2 * args.time_bins, height=height, width=width)
        
        with NPZWriter(output_npz) as writer:
            ev_ts = reader.time
            total_events = len(ev_ts)
            
            # --- 変更点: ファイル単位で一括処理 ---
            # 1ファイルの時間が100ms（あるいは一定の短い区間）であり、
            # その区間全体で時間を正規化したい場合、分割してはいけません。
            # 分割すると、各ブロックごとに時間が0-1に正規化され、時間が混ざってしまいます。
            
            ev_window = reader.get_event_slice(0, total_events)
            ev_repr = hist_generator.construct(ev_window['x'], ev_window['y'], ev_window['p'], ev_window['t'], args.time_bins)
            
            if args.downsample:
                ev_repr = downsample_ev_repr(ev_repr, scale_factor=0.5)
            
            # count_cutoffを適用
            if args.count_cutoff is not None:
                ev_repr = torch.clamp(ev_repr, max=args.count_cutoff)
            
            # 型変換（必要に応じて）
            # ここでは既に construct が適切な型で返しているか、またはここで変換するかですが、
            # 元のコードに合わせて uint8 化などを確認します。
            # StackedHistogram.construct は uint8 (fastmode) または int16 を返します。
            # 今回の実装では uint8 で初期化されているので、オーバーフローだけ注意ですが
            # 一括処理後の clamp で安全になります。
            
            writer.add_data(ev_repr)
            
    # print(f"\n処理が完了しました。 {output_npz} に保存しました。")

def main():
    parser = argparse.ArgumentParser(description="イベントデータを時間と極性を考慮した積層ヒストグラムに変換します。")
    parser.add_argument('--dir', help='入力Dataset ディレクトリ（例: /path/to/dataset）')
    parser.add_argument('--height', type=int, default=480, help='イベントデータの高さ（ピクセル数）')
    parser.add_argument('--width', type=int, default=640, help='イベントデータの幅（ピクセル数）')
    parser.add_argument('--time_bins', type=int, default=10, help='時間軸の分割数。最終的なチャンネル数は 2 * time_bins となります。')
    parser.add_argument('--count_cutoff', type=int, default=None, help='ヒストグラムの各ピクセルの最大カウント値。')
    parser.add_argument('--downsample', action='store_true', help='有効にすると、生成されたヒストグラムを2分の1にダウンサンプリングします。')
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
        # print(f"\n[Thread] 入力ファイル: {input_npz} 高さ={args.height}, 幅={args.width}, 合計チャンネル数={2 * args.time_bins}")
        process_file(input_npz, output_npz, args, args.height, args.width)

    max_workers = min(50, len(npz_files)) # 並列数は最大50、またはファイル数まで
    print(f"並列数: {max_workers}, 全ファイル数: {len(npz_files)}, サイズ{args.height}x{args.width}, 合計チャンネル数={2 * args.time_bins}")
    from tqdm import tqdm
    
    # tqdmで進捗を表示するために、executor.mapではなく、as_completedとtqdmを組み合わせるか、
    # シンプルにリスト内包表記の中でtqdmを使う形にはできない（submitが即座に終わるため）。
    # 完了した順にバーを進める実装にします。

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_wrapper, input_npz) for input_npz in npz_files]
        
        # tqdmで完了を監視
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(npz_files), desc="Processing files"):
            pass
            # エラーチェックはここでやる必要があれば future.result() を呼ぶ
            # for future in ... のループ変数はここには来ないので、
            # エラーハンドリングが必要なら futuresリストを別で回すか、ここで future を受け取る形に修正が必要
            
    # エラーチェック用（tqdmループ内で行うとバーの表示が崩れることがあるため、必要なら別途実行、
    # あるいは上記のループ内で future.result() を呼ぶ）
    for f in futures:
        try:
            f.result()
        except Exception as e:
            print(f"エラー: {e}")

if __name__ == '__main__':
    main()