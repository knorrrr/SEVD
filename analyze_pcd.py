import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import json

class LidarDistanceAnalyzer:
    """Analyze LiDAR data by distance ranges"""
    
    def __init__(self, base_dir: str, distance_bins: List[float] = None, 
                 horizontal_fov: float = 360.0, vertical_fov: float = 26.9):
        """
        Args:
            base_dir: Base directory to analyze (e.g., 11_towns_each_6000_ticks_20251123_215345)
            distance_bins: Distance bin boundaries (default: [0, 10, 20, 30, 40, 50, 100])
            horizontal_fov: Horizontal field of view in degrees (default: 360.0 for full rotation)
            vertical_fov: Vertical field of view in degrees (default: 26.9 for typical LiDAR)
        """
        self.base_dir = Path(base_dir)
        self.distance_bins = distance_bins if distance_bins else [0, 10, 20, 30, 40, 50, 100]
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.results = defaultdict(lambda: defaultdict(list))
        
    def find_lidar_directories(self) -> List[Path]:
        """Find all lidar-front_filtered_downsampled directories"""
        lidar_dirs = []
        for town_dir in sorted(self.base_dir.glob("Town*")):
            if town_dir.is_dir():
                lidar_path = town_dir / "ego0" / "lidar-front_filtered_downsampled"
                if lidar_path.exists():
                    lidar_dirs.append(lidar_path)
        return lidar_dirs
    
    def load_point_cloud(self, bin_file: Path) -> np.ndarray:
        """
        Load point cloud data from .bin file
        
        Returns:
            points: (N, 4) array [x, y, z, intensity]
        """
        try:
            points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
            return points
        except Exception as e:
            print(f"Warning: Failed to load {bin_file.name} - {e}")
            return np.array([]).reshape(0, 4)
    
    def calculate_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Calculate distance from origin for each point
        
        Args:
            points: (N, 4) array [x, y, z, intensity]
            
        Returns:
            distances: (N,) array
        """
        return np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
    
    def analyze_distance_distribution(self, points: np.ndarray) -> Dict[str, int]:
        """
        Calculate point distribution by distance ranges
        
        Returns:
            bin_counts: Point count for each distance range
        """
        if len(points) == 0:
            return {self.get_bin_label(i): 0 for i in range(len(self.distance_bins) - 1)}
        
        distances = self.calculate_distances(points)
        bin_counts = {}
        
        for i in range(len(self.distance_bins) - 1):
            min_dist = self.distance_bins[i]
            max_dist = self.distance_bins[i + 1]
            count = np.sum((distances >= min_dist) & (distances < max_dist))
            bin_counts[self.get_bin_label(i)] = int(count)
        
        return bin_counts
    
    def get_bin_label(self, bin_index: int) -> str:
        """Generate bin label"""
        min_dist = self.distance_bins[bin_index]
        max_dist = self.distance_bins[bin_index + 1]
        return f"{min_dist:.0f}-{max_dist:.0f}m"
    
    def calculate_sector_volume(self, r_min: float, r_max: float) -> float:
        """
        Calculate volume of a spherical sector (cone-shaped region)
        
        Args:
            r_min: Inner radius (meters)
            r_max: Outer radius (meters)
            
        Returns:
            Volume in cubic meters
        """
        # Convert FOV from degrees to radians
        theta_h = np.radians(self.horizontal_fov)
        theta_v = np.radians(self.vertical_fov)
        
        # Volume of spherical sector = (2/3) * π * (r_outer³ - r_inner³) * (1 - cos(θ_v/2)) * (θ_h / 2π)
        # Simplified: (1/3) * (r_outer³ - r_inner³) * θ_h * (1 - cos(θ_v/2))
        
        solid_angle = theta_h * (1 - np.cos(theta_v / 2))
        volume = (1.0 / 3.0) * (r_max**3 - r_min**3) * solid_angle
        
        return volume
    
    def analyze_directory(self, lidar_dir: Path) -> Dict:
        """
        Analyze one LiDAR directory
        
        Returns:
            summary: Dictionary of statistics
        """
        bin_files = sorted(lidar_dir.glob("*.bin"))
        
        if not bin_files:
            print(f"警告: {lidar_dir} に.binファイルが見つかりません")
            return {}
        
        # Aggregate statistics across all files
        total_points = 0
        total_bin_counts = defaultdict(int)
        file_count = 0
        
        print(f"  分析中: {lidar_dir.relative_to(self.base_dir)} ({len(bin_files)} ファイル)")
        
        for bin_file in bin_files:
            points = self.load_point_cloud(bin_file)
            if len(points) == 0:
                continue
                
            total_points += len(points)
            bin_counts = self.analyze_distance_distribution(points)
            
            for bin_label, count in bin_counts.items():
                total_bin_counts[bin_label] += count
            
            file_count += 1
        
        # Calculate averages
        avg_points = total_points / file_count if file_count > 0 else 0
        avg_bin_counts = {k: v / file_count for k, v in total_bin_counts.items()}
        
        # Calculate percentages
        bin_percentages = {}
        if total_points > 0:
            bin_percentages = {k: (v / total_points) * 100 for k, v in total_bin_counts.items()}
        
        # Calculate point density (points per cubic meter)
        bin_densities = {}
        avg_bin_densities = {}
        for i in range(len(self.distance_bins) - 1):
            bin_label = self.get_bin_label(i)
            r_min = self.distance_bins[i]
            r_max = self.distance_bins[i + 1]
            volume = self.calculate_sector_volume(r_min, r_max)
            
            if volume > 0:
                bin_densities[bin_label] = total_bin_counts[bin_label] / volume
                avg_bin_densities[bin_label] = avg_bin_counts[bin_label] / volume
            else:
                bin_densities[bin_label] = 0
                avg_bin_densities[bin_label] = 0
        
        return {
            "total_files": file_count,
            "total_points": total_points,
            "avg_points_per_file": avg_points,
            "total_bin_counts": dict(total_bin_counts),
            "avg_bin_counts": avg_bin_counts,
            "bin_percentages": bin_percentages,
            "bin_densities": bin_densities,
            "avg_bin_densities": avg_bin_densities,
            "path": str(lidar_dir.relative_to(self.base_dir))
        }
    
    def analyze_all(self) -> Dict:
        """Analyze all LiDAR directories"""
        lidar_dirs = self.find_lidar_directories()
        
        if not lidar_dirs:
            print(f"エラー: {self.base_dir} にLiDARディレクトリが見つかりません")
            return {}
        
        print(f"\n{len(lidar_dirs)} 個のLiDARディレクトリを検出しました\n")
        
        all_results = {}
        for lidar_dir in lidar_dirs:
            town_name = lidar_dir.parent.parent.name
            result = self.analyze_directory(lidar_dir)
            if result:
                all_results[town_name] = result
        
        return all_results
    
    def save_text_report(self, results: Dict, output_path: Path):
        """Save text report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LiDAR点群 距離別分析レポート\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"ベースディレクトリ: {self.base_dir}\n")
            f.write(f"分析対象: {len(results)} Towns\n")
            f.write(f"距離ビン: {self.distance_bins}\n")
            f.write(f"水平視野角: {self.horizontal_fov}°\n")
            f.write(f"垂直視野角: {self.vertical_fov}°\n\n")
            
            for town_name, data in sorted(results.items()):
                f.write("-" * 80 + "\n")
                f.write(f"Town: {town_name}\n")
                f.write(f"パス: {data['path']}\n")
                f.write(f"ファイル数: {data['total_files']}\n")
                f.write(f"総点数: {data['total_points']:,}\n")
                f.write(f"平均点数/ファイル: {data['avg_points_per_file']:.1f}\n\n")
                
                f.write("距離別分布:\n")
                f.write(f"  {'距離範囲':<15} {'総点数':<15} {'平均点数/ファイル':<20} {'割合':<10}\n")
                f.write("  " + "-" * 60 + "\n")
                
                for bin_label in sorted(data['total_bin_counts'].keys()):
                    total = data['total_bin_counts'][bin_label]
                    avg = data['avg_bin_counts'][bin_label]
                    pct = data['bin_percentages'][bin_label]
                    f.write(f"  {bin_label:<15} {total:<15,} {avg:<20.1f} {pct:>6.2f}%\n")
                
                f.write("\n")
                
                # Add density information
                f.write("点群密度 (points/m³):\n")
                f.write(f"  {'距離範囲':<15} {'累積密度(全フレーム)':<25} {'1フレームあたりの密度':<25}\n")
                f.write("  " + "-" * 65 + "\n")
                
                for bin_label in sorted(data['bin_densities'].keys()):
                    total_density = data['bin_densities'][bin_label]
                    avg_density = data['avg_bin_densities'][bin_label]
                    f.write(f"  {bin_label:<15} {total_density:<25.2f} {avg_density:<25.2f}\n")
                
                f.write("\n")
            
            # Overall statistics
            f.write("=" * 80 + "\n")
            f.write("全体統計\n")
            f.write("=" * 80 + "\n")
            
            total_files = sum(d['total_files'] for d in results.values())
            total_points = sum(d['total_points'] for d in results.values())
            
            f.write(f"総ファイル数: {total_files:,}\n")
            f.write(f"総点数: {total_points:,}\n")
            f.write(f"平均点数/ファイル: {total_points / total_files:.1f}\n\n")
            
            # Overall distance distribution
            overall_bin_counts = defaultdict(int)
            for data in results.values():
                for bin_label, count in data['total_bin_counts'].items():
                    overall_bin_counts[bin_label] += count
            
            f.write("全体の距離別分布:\n")
            f.write(f"  {'距離範囲':<15} {'総点数':<15} {'割合':<10}\n")
            f.write("  " + "-" * 40 + "\n")
            
            for bin_label in sorted(overall_bin_counts.keys()):
                count = overall_bin_counts[bin_label]
                pct = (count / total_points) * 100 if total_points > 0 else 0
                f.write(f"  {bin_label:<15} {count:<15,} {pct:>6.2f}%\n")
            
            f.write("\n")
            
            # Overall density
            f.write("全体の点群密度 (points/m³):\n")
            f.write(f"  {'距離範囲':<15} {'累積密度(全フレーム)':<25} {'1フレームあたりの密度':<25}\n")
            f.write("  " + "-" * 65 + "\n")
            
            for i in range(len(self.distance_bins) - 1):
                bin_label = self.get_bin_label(i)
                r_min = self.distance_bins[i]
                r_max = self.distance_bins[i + 1]
                volume = self.calculate_sector_volume(r_min, r_max)
                
                total_density = overall_bin_counts[bin_label] / volume if volume > 0 else 0
                avg_density = (overall_bin_counts[bin_label] / total_files) / volume if volume > 0 else 0
                f.write(f"  {bin_label:<15} {total_density:<25.2f} {avg_density:<25.2f}\n")
        
        print(f"\nテキストレポートを保存: {output_path}")
    
    def save_json_report(self, results: Dict, output_path: Path):
        """Save report in JSON format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSONレポートを保存: {output_path}")
    
    def plot_results(self, results: Dict, output_dir: Path):
        """Visualize analysis results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Distance distribution by town (stacked bar chart)
        self._plot_distance_distribution_by_town(results, output_dir / "distance_distribution_by_town.png")
        
        # 2. Overall distance distribution (pie chart)
        self._plot_overall_distance_distribution(results, output_dir / "overall_distance_distribution.png")
        
        # 3. Average points comparison by town
        self._plot_avg_points_comparison(results, output_dir / "avg_points_comparison.png")
        
        # 4. Distance distribution boxplot
        self._plot_distance_boxplot(results, output_dir / "distance_boxplot.png")
        
        print(f"\n可視化結果を保存: {output_dir}")
    
    def _plot_distance_distribution_by_town(self, results: Dict, output_path: Path):
        """Plot distance distribution by town as stacked bar chart"""
        towns = sorted(results.keys())
        bin_labels = sorted(list(results[towns[0]]['bin_percentages'].keys()))
        
        # Prepare data
        data = {bin_label: [] for bin_label in bin_labels}
        for town in towns:
            for bin_label in bin_labels:
                data[bin_label].append(results[town]['bin_percentages'][bin_label])
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        x = np.arange(len(towns))
        width = 0.6
        bottom = np.zeros(len(towns))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_labels)))
        
        for i, (bin_label, color) in enumerate(zip(bin_labels, colors)):
            ax.bar(x, data[bin_label], width, label=bin_label, bottom=bottom, color=color)
            bottom += data[bin_label]
        
        ax.set_xlabel('Town', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Point Cloud Distribution by Distance Range (Each Town)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(towns, rotation=45, ha='right')
        ax.legend(title='Distance Range', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_overall_distance_distribution(self, results: Dict, output_path: Path):
        """Plot overall distance distribution as pie chart"""
        overall_bin_counts = defaultdict(int)
        for data in results.values():
            for bin_label, count in data['total_bin_counts'].items():
                overall_bin_counts[bin_label] += count
        
        bin_labels = sorted(overall_bin_counts.keys())
        counts = [overall_bin_counts[label] for label in bin_labels]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_labels)))
        
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=bin_labels, 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            textprops={'fontsize': 11}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Overall Point Cloud Distribution by Distance', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_avg_points_comparison(self, results: Dict, output_path: Path):
        """Compare average points per file across towns"""
        towns = sorted(results.keys())
        avg_points = [results[town]['avg_points_per_file'] for town in towns]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(towns, avg_points, color='skyblue', edgecolor='navy', alpha=0.7)
        
        ax.set_xlabel('Town', fontsize=12)
        ax.set_ylabel('Average Points per File', fontsize=12)
        ax.set_title('Average Point Cloud Size Comparison', fontsize=14, fontweight='bold')
        ax.set_xticklabels(towns, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Display values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distance_boxplot(self, results: Dict, output_path: Path):
        """Plot distance distribution as boxplot"""
        bin_labels = sorted(list(results[list(results.keys())[0]]['avg_bin_counts'].keys()))
        
        # Prepare data
        data = {bin_label: [] for bin_label in bin_labels}
        for town_data in results.values():
            for bin_label in bin_labels:
                data[bin_label].append(town_data['avg_bin_counts'][bin_label])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        positions = np.arange(len(bin_labels))
        bp = ax.boxplot([data[label] for label in bin_labels], 
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showmeans=True)
        
        # Set colors
        colors = plt.cm.viridis(np.linspace(0, 1, len(bin_labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Distance Range', fontsize=12)
        ax.set_ylabel('Average Points per File', fontsize=12)
        ax.set_title('Point Distribution by Distance Range (All Towns)', fontsize=14, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("使い方:")
        print("  python analyze_pcd.py <ベースディレクトリ名>")
        print("\n例:")
        print("  python analyze_pcd.py 11_towns_each_6000_ticks_20251123_215345")
        print("\nオプション:")
        print("  --bins <距離1> <距離2> ... : カスタム距離ビンを指定")
        print("    例: --bins 0 15 30 45 60 100")
        return
    
    # Parse arguments
    base_dir_name = sys.argv[1]
    base_dir = Path("/media/ssd/SEVD/carla/out") / base_dir_name
    
    if not base_dir.exists():
        print(f"エラー: ディレクトリが見つかりません: {base_dir}")
        return
    
    # Handle custom distance bins
    distance_bins = None
    if "--bins" in sys.argv:
        bins_idx = sys.argv.index("--bins")
        try:
            distance_bins = [float(x) for x in sys.argv[bins_idx + 1:]]
            print(f"カスタム距離ビンを使用: {distance_bins}")
        except (ValueError, IndexError):
            print("警告: 距離ビンの指定が不正です。デフォルト値を使用します。")
    
    # Run analysis
    analyzer = LidarDistanceAnalyzer(base_dir, distance_bins)
    results = analyzer.analyze_all()
    
    if not results:
        print("分析結果がありません。")
        return
    
    # Create output directory
    output_dir = base_dir / "lidar_analysis"
    output_dir.mkdir(exist_ok=True)
    
    # Save reports
    analyzer.save_text_report(results, output_dir / "analysis_report.txt")
    analyzer.save_json_report(results, output_dir / "analysis_report.json")
    
    # Visualize
    analyzer.plot_results(results, output_dir / "plots")
    
    print("\n" + "=" * 80)
    print("分析完了!")
    print(f"結果は以下に保存されました: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()