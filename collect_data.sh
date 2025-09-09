#!/bin/bash

# スクリプトが失敗した場合に途中で終了させる
set -e

# --- 変数定義 (ご自身の環境に合わせて修正してください) ---
CARLA_SIM_DIR="/media/ssd/SEVD/carla"
CARLA_UE4_DIR="~/carla"
CARLA_CONDA_ENV="carla"
POINTCEPT_CONDA_ENV="pointcept-torch2.5.0-cu12.4"
OUTPUT_BASE_DIR="$CARLA_SIM_DIR/out" # CARLAの出力先ベースディレクトリ

# --- 1. CARLAシミュレーションの実行 ---
echo "--- Starting CARLA Simulation ---"
eval cd "$CARLA_UE4_DIR"
./CarlaUE4.sh &
CARLA_PID=$!
echo "CARLA simulator started with PID: $CARLA_PID"
echo "Waiting 15 seconds for CARLA to initialize..."
sleep 15

cd "$CARLA_SIM_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CARLA_CONDA_ENV"

echo "Running data generation script (main.py)..."
python main.py \
    --number-of-ego-vehicles=1 \
    -n=120 \
    -w=70 \
    --sync \
    --delta-seconds=0.1 \
    --timeout=60 \
    --ignore-first-n-ticks=35 \
    --duration=5 \
    --start-weather=ClearNoon \
    --end-weather=ClearNight || true

echo "Data generation finished."

# --- 2. 後処理の実行 ---
echo -e "\n--- Starting Post-Processing ---"

# 最新の出力ディレクトリを自動で検索して取得
echo "Finding the latest output directory..."
LATEST_RUN_DIR=$(find "$OUTPUT_BASE_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

if [ -z "$LATEST_RUN_DIR" ]; then
    echo "Error: Could not find any output directory in $OUTPUT_BASE_DIR"
    kill $CARLA_PID # エラー時はCARLAを終了
    exit 1
fi

EGO_DIR="$LATEST_RUN_DIR/ego0"
echo "Processing directory: $EGO_DIR"

# 後処理用のConda環境をアクティベート
echo "Activating conda environment: $POINTCEPT_CONDA_ENV"
conda activate "$POINTCEPT_CONDA_ENV"

# 各種前処理スクリプトを実行
echo "Running preprocess_ev.py..."
cd ../
python3 preprocess_projection_point.py --input-dir "$EGO_DIR"
python3 preprocess_ev.py --dir "$EGO_DIR"

echo "Running preprocess_evpcd.py..."
python3 preprocess_evpcd.py "$EGO_DIR"

echo "Post-processing finished."

# --- 3. CARLAシミュレータの終了 ---
echo -e "\n--- Shutting Down ---"
echo "Shutting down CARLA simulator (PID: $CARLA_PID)..."
kill $CARLA_PID
echo "All tasks completed successfully."