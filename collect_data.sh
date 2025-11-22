#!/bin/bash

# スクリプトが失敗した場合に途中で終了させる
set -e

# --- 変数定義 (ご自身の環境に合わせて修正してください) ---
SEVD_DIR="/media/ssd/SEVD"
CARLA_UE4_DIR="~/carla"
CARLA_CONDA_ENV="carla0915"
POINTCEPT_CONDA_ENV="pointcept-torch2.5.0-cu12.8"
OUTPUT_BASE_DIR="$SEVD_DIR/carla/out" # CARLAの出力先ベースディレクトリ

# --- 1. マップリストの定義 ---
# MAPS=("Town01_Opt" "Town02_Opt" "Town03_Opt" "Town04_Opt" "Town05_Opt" "Town06" "Town07" "Town10HD_Opt" "Town11" "Town12" "Town13" "Town15")
MAPS=("Town01_Opt" "Town02_Opt")
ALL_EGO_DIRS=()
DURATION=5

# --- 出力ディレクトリの構成 ---
NUM_TOWNS=${#MAPS[@]}
PARENT_OUTPUT_DIR_NAME="${NUM_TOWNS}_towns_each_${DURATION}_ticks_$(date +%Y%m%d_%H%M%S)"
PARENT_OUTPUT_DIR="$OUTPUT_BASE_DIR/$PARENT_OUTPUT_DIR_NAME"
echo "Output will be saved to: $PARENT_OUTPUT_DIR"

# --- 既存のCARLAプロセスを終了 ---
echo "Checking for existing CARLA processes..."
if pgrep -f "CarlaUE4" > /dev/null; then
    echo "Found running CARLA process. Killing it..."
    pkill -f "CarlaUE4"
    sleep 5 # プロセスが完全に終了するのを待つ
    echo "CARLA process killed."
else
    echo "No existing CARLA process found."
fi

echo "--- Starting CARLA Simulation for $MAP_NAME ---"
eval cd "$CARLA_UE4_DIR"
./CarlaUE4.sh &
CARLA_PID=$!
echo "CARLA simulator started with PID: $CARLA_PID"
echo "Waiting 15 seconds for CARLA to initialize..."
sleep 20
# --- 2. 各マップでのデータ収集と個別前処理 ---
for MAP_NAME in "${MAPS[@]}"; do
    echo "=================================================="
    echo "Processing Map: $MAP_NAME"
    echo "=================================================="
    # --- CARLAシミュレーションの実行 ---
    cd "$SEVD_DIR/carla"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CARLA_CONDA_ENV"

    echo "Running data generation script (main.py) for $MAP_NAME..."
    python main.py \
        --map="$MAP_NAME" \
        --number-of-ego-vehicles=1 \
        -n=120 \
        -w=70 \
        --sync \
        --delta-seconds=0.1 \
        --timeout=60 \
        --ignore-first-n-ticks=1 \
        --duration=$DURATION \
        --output-dir="$PARENT_OUTPUT_DIR" \
        --start-weather=ClearNoon \
        --end-weather=ClearNight || true
        # --ignore-first-n-ticks=35 \

    echo "Data generation finished for $MAP_NAME."

    
    # --- 後処理の準備 ---
    echo "Finding the latest output directory for $MAP_NAME..."
    # 最新の出力ディレクトリを自動で検索して取得
    LATEST_RUN_DIR=$(find "$PARENT_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "${MAP_NAME}_*" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

    if [ -z "$LATEST_RUN_DIR" ]; then
        echo "Error: Could not find any output directory in $PARENT_OUTPUT_DIR for $MAP_NAME"
        continue # 次のマップへ
    fi

    EGO_DIR="$LATEST_RUN_DIR/ego0"
    echo "Processing directory: $EGO_DIR"
    ALL_EGO_DIRS+=("$EGO_DIR")

    # --- 個別前処理の実行 ---
    echo "Activating conda environment: $POINTCEPT_CONDA_ENV"
    conda activate "$POINTCEPT_CONDA_ENV"

    echo "Running preprocess_ev.py..."
    cd "$SEVD_DIR" # 元のディレクトリに戻る (念のため)
    python3 preprocess_projection_point.py --input-dir "$EGO_DIR"
    python3 preprocess_ev.py --dir "$EGO_DIR"

done

# --- CARLAシミュレータの終了 ---
echo "Shutting down CARLA simulator (PID: $CARLA_PID)..."
kill $CARLA_PID

# --- 3. 統合後処理の実行 ---
echo -e "\n--- Starting Aggregated Post-Processing ---"
if [ ${#ALL_EGO_DIRS[@]} -eq 0 ]; then
    echo "Error: No data collected from any map."
    exit 1
fi
echo "Running pcd_downsample.py with all collected directories..."
python3 pcd_downsample.py "${ALL_EGO_DIRS[@]}"

echo "Running preprocess_evpcd.py with all collected directories..."
echo "${ALL_EGO_DIRS[@]}"
python3 preprocess_evpcd.py --duration $DURATION "${ALL_EGO_DIRS[@]}"

echo "All tasks completed successfully."