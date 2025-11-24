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
# 11 Maps
MAPS=("Town01_Opt" "Town02_Opt" "Town03" "Town04_Opt" "Town05_Opt" "Town06" "Town07" "Town10HD_Opt" "Town12" "Town13" "Town15")
# Large
# MAPS=("Town12" "Town13")
ALL_EGO_DIRS=()
ALL_DURATIONS=()
# DURATION=6000
# LARGE_MAP_DURATION=12000 # Duration for Town12 and Town13

DURATION=6000
LARGE_MAP_DURATION=12000 # Duration for Town12 and Town13
# --- 出力ディレクトリの構成 ---
NUM_TOWNS=${#MAPS[@]}
PARENT_OUTPUT_DIR_NAME="${NUM_TOWNS}_towns_each_${DURATION}_ticks_$(date +%Y%m%d_%H%M%S)"
PARENT_OUTPUT_DIR="$OUTPUT_BASE_DIR/$PARENT_OUTPUT_DIR_NAME"
echo "Output will be saved to: $PARENT_OUTPUT_DIR"

# --- 2. 各マップでのデータ収集と個別前処理 ---
for MAP_NAME in "${MAPS[@]}"; do
    echo " "
    echo "=================================================="
    echo "Processing Map: $MAP_NAME"
    echo "=================================================="

    # Determine duration for this map
    if [[ "$MAP_NAME" == *"Town12"* ]] || [[ "$MAP_NAME" == *"Town13"* ]]; then
        CURRENT_DURATION=$LARGE_MAP_DURATION
        echo "Using Large Map Duration: $CURRENT_DURATION"
    else
        CURRENT_DURATION=$DURATION
        echo "Using Standard Duration: $CURRENT_DURATION"
    fi

    MAX_RETRIES=5
    RETRY_COUNT=0
    SUCCESS=false
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        echo "Attempt $((RETRY_COUNT+1)) of $MAX_RETRIES for $MAP_NAME"

        # --- 既存のCARLAプロセスと関連ポートのクリーンアップ ---
        echo "Cleaning up processes..."
        
        # Kill CarlaUE4
        if pgrep -f "CarlaUE4" > /dev/null; then
            echo "Found running CARLA process. Killing it..."
            pkill -9 -f "CarlaUE4"
        fi

        # Kill processes on ports 2000-2002 (CARLA) and 8000 (Traffic Manager)
        echo "Killing processes on ports 2000-2002 and 8000..."
        fuser -k -9 2000/tcp >/dev/null 2>&1 || true
        fuser -k -9 2001/tcp >/dev/null 2>&1 || true
        fuser -k -9 2002/tcp >/dev/null 2>&1 || true
        fuser -k -9 8000/tcp >/dev/null 2>&1 || true
        
        sleep 5

        # --- CARLAシミュレーションの起動 ---
        echo "=================================================="
        echo "Starting CARLA Simulation..."
        echo "=================================================="
        eval cd "$CARLA_UE4_DIR"
        ./CarlaUE4.sh -RenderOffScreen &
        CARLA_PID=$!
        echo "CARLA simulator started with PID: $CARLA_PID"
        if [[ "$MAP_NAME" == *"Town13"* ]]; then
            CARLA_INIT_WAIT=60
        else
            CARLA_INIT_WAIT=20
        fi
        echo "Waiting $CARLA_INIT_WAIT seconds for CARLA to initialize..."
        sleep $CARLA_INIT_WAIT

        
        # Check if CARLA is still running
        if ! kill -0 $CARLA_PID 2>/dev/null; then
            echo "Error: CARLA simulator crashed during initialization (PID: $CARLA_PID)."
            echo "Retrying..."
            RETRY_COUNT=$((RETRY_COUNT+1))
            continue
        fi
        
        # --- CARLAシミュレーションの実行 ---
        cd "$SEVD_DIR/carla"
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CARLA_CONDA_ENV"

        echo "=================================================="
        echo "Running data generation script (main.py) for $MAP_NAME..."
        echo "=================================================="
        
        # set +e を使ってエラーでもスクリプトが止まらないようにする
        # Duration is frame ticks
        set +e
        python main.py \
            --map="$MAP_NAME" \
            --number-of-ego-vehicles=1 \
            -n=120 \
            -w=70 \
            --sync \
            --delta-seconds=0.1 \
            --timeout=120 \
            --ignore-first-n-ticks=0 \
            --duration=$CURRENT_DURATION \
            --output-dir="$PARENT_OUTPUT_DIR" \
            --start-weather=ClearNoon \
            --end-weather=ClearNight
        
        EXIT_CODE=$?
        set -e

        # 最新の出力ディレクトリを取得 (main.pyが作成したもの)
        LATEST_RUN_DIR=$(find "$PARENT_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "${MAP_NAME}_*" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
        
        DATA_GENERATED=false
        if [ -n "$LATEST_RUN_DIR" ]; then
             # metadataファイルが存在し、かつLiDARデータ(.bin)が生成されているか確認
             if ls "$LATEST_RUN_DIR"/metadata-*.json 1> /dev/null 2>&1; then
                 if [ -d "$LATEST_RUN_DIR/ego0/lidar-front" ] && find "$LATEST_RUN_DIR/ego0/lidar-front" -name "*.bin" | grep -q .; then
                     DATA_GENERATED=true
                 fi
             fi
        fi

        # 0 (正常終了) または 134 (SIGABRT/Crash on exit) かつ、データが生成されている場合を成功とみなす
        if { [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 134 ]; } && [ "$DATA_GENERATED" = true ]; then
            if [ $EXIT_CODE -eq 0 ]; then
                echo "Data generation finished successfully for $MAP_NAME."
            else
                echo "Data generation finished with exit code $EXIT_CODE, and data was generated. Treating as success for $MAP_NAME."
            fi
            SUCCESS=true
            
            # 成功したらCARLAを終了してループを抜ける
            echo "Shutting down CARLA simulator (PID: $CARLA_PID)..."
            kill $CARLA_PID 2>/dev/null || true
            break
        else
            if [ "$DATA_GENERATED" = false ]; then
                echo "Error: main.py exited with code $EXIT_CODE but NO valid data (metadata/lidar) was generated."
                # Remove the failed directory to keep output clean
                if [ -n "$LATEST_RUN_DIR" ] && [ -d "$LATEST_RUN_DIR" ]; then
                    echo "Removing failed output directory: $LATEST_RUN_DIR"
                    rm -rf "$LATEST_RUN_DIR"
                fi
            else
                echo "Error: main.py failed with exit code $EXIT_CODE."
                # If data was generated but exit code is bad, we might want to keep it or delete it.
                # Usually if exit code is bad, we assume failure. But logic above says if data generated, we treat as success?
                # Ah, line 108 handles success. This else block is for FAILURE.
                # So if we are here, it means either (ExitCode != 0 AND != 134) OR (DataGenerated == false).
                
                # If DataGenerated is true but ExitCode is bad (and not 134), we probably should keep it for inspection or delete it?
                # User asked to delete "failed works".
                # Let's delete it if we are retrying.
                if [ -n "$LATEST_RUN_DIR" ] && [ -d "$LATEST_RUN_DIR" ]; then
                    echo "Removing failed output directory (bad exit code): $LATEST_RUN_DIR"
                    rm -rf "$LATEST_RUN_DIR"
                fi
            fi
            echo "Shutting down CARLA simulator (PID: $CARLA_PID) and retrying..."
            kill $CARLA_PID 2>/dev/null || true
            RETRY_COUNT=$((RETRY_COUNT+1))
            sleep 5
        fi
    done

    if [ "$SUCCESS" = false ]; then
        echo "Error: Failed to process $MAP_NAME after $MAX_RETRIES attempts. Skipping..."
        continue
    fi

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

    echo "=================================================="
    echo "Running preprocess_ev.py..."
    echo "=================================================="
    cd "$SEVD_DIR" # 元のディレクトリに戻る (念のため)
    python3 preprocess_projection_point.py --input-dir "$EGO_DIR"
    python3 preprocess_ev.py --dir "$EGO_DIR"
    
    # Store duration for this run
    ALL_DURATIONS+=("$CURRENT_DURATION")
done

# --- 3. 統合後処理の実行 ---
echo -e "\n--- Starting Aggregated Post-Processing ---"
if [ ${#ALL_EGO_DIRS[@]} -eq 0 ]; then
    echo "Error: No data collected from any map."
    exit 1
fi
echo "=================================================="
echo "Running preprocess_pcd_downsample.py with all collected directories..."
echo "=================================================="
python3 preprocess_pcd_downsample.py "${ALL_EGO_DIRS[@]}"

echo "=================================================="
echo "Running preprocess_evpcd.py with all collected directories..."
echo "=================================================="
echo "Directories: ${ALL_EGO_DIRS[@]}"
echo "Durations: ${ALL_DURATIONS[@]}"
python3 preprocess_evpcd.py "${ALL_EGO_DIRS[@]}" --durations "${ALL_DURATIONS[@]}"

echo "All tasks completed successfully."