#!/usr/bin/env bash
set -euo pipefail
BASE=/home/zhanglesong/can_anomaly_detection
REPO=$BASE/detector_experiment
OUT=$BASE/reference_bank_full
COL=$REPO/third_party/colmap
export CUDA_HOME=/home/zhanglesong/.local/lib/python3.10/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO/third_party/gaussian-splatting:$REPO/third_party/sam2"
export LD_LIBRARY_PATH="$COL/usr/lib:$COL/usr/lib/x86_64-linux-gnu:$COL/usr/lib/x86_64-linux-gnu/lapack:$COL/usr/lib/x86_64-linux-gnu/atlas:$COL/usr/lib/x86_64-linux-gnu/blas:/home/zhanglesong/.local/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
export QT_PLUGIN_PATH="$COL/usr/lib/x86_64-linux-gnu/qt5/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COL/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
mkdir -p "$OUT/logs"
cd "$REPO"
common=(--video-root "$BASE/data/04_输入数据/原始输入数据" --output-root "$OUT" --gaussian-repo "$REPO/third_party/gaussian-splatting" --sam2-repo "$REPO/third_party/sam2" --sam-checkpoint "$BASE/models/sam2.1_hiera_b_plus/sam2.1_hiera_base_plus.pt" --sam-config "$REPO/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml" --reference-count 720 --fps 1 --iterations 30000 --colmap "$COL/usr/bin/colmap" --colmap-no-gpu)
run_one() {
  local brand="$1" gpu="$2" port="$3" logfile="$4"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python3 build_reference_bank.py "${common[@]}" --brands "$brand" --gpu "$gpu" --port "$port" > "$OUT/logs/$logfile" 2>&1
}
run_one 哈尔滨清爽 0 6010 harbin_qingshuang.log & p1=$!
run_one 嘉士伯 1 6011 jiashibo.log & p2=$!
run_one 科罗娜啤酒 3 6013 keluona.log & p3=$!
status=0
wait "$p1" || status=1
wait "$p2" || status=1
wait "$p3" || status=1
if [ "$status" -ne 0 ]; then exit "$status"; fi
python3 build_reference_bank.py "${common[@]}" --brands 哈尔滨清爽 嘉士伯 科罗娜啤酒 --gpu 0 --port 6014 --reuse > "$OUT/logs/aggregate.log" 2>&1
