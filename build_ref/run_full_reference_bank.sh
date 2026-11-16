#!/usr/bin/env bash
set -euo pipefail
BASE=/home/zhanglesong/can_anomaly_detection
REPO=$BASE/detector_experiment
MODEL=$BASE/models/3dgs
REF=$BASE/data/reference
COL=$REPO/third_party/colmap
PY=$REPO/.venv/bin/python
export CUDA_HOME=$REPO/.venv/lib/python3.10/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO/third_party/gaussian-splatting:$REPO/third_party/sam2"
export LD_LIBRARY_PATH="$COL/usr/lib:$COL/usr/lib/x86_64-linux-gnu:$COL/usr/lib/x86_64-linux-gnu/lapack:$COL/usr/lib/x86_64-linux-gnu/atlas:$COL/usr/lib/x86_64-linux-gnu/blas:$REPO/.venv/lib/python3.10/site-packages/torch/lib:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM=offscreen
export QT_PLUGIN_PATH="$COL/usr/lib/x86_64-linux-gnu/qt5/plugins"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COL/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"
mkdir -p "$MODEL/logs" "$REF"
cd "$REPO/build_ref"
common=(--video-root "$BASE/data/04_输入数据/00_raw/reference_inputs" --model-root "$MODEL" --references-root "$REF" --gaussian-repo "$REPO/third_party/gaussian-splatting" --sam2-repo "$REPO/third_party/sam2" --sam-checkpoint "$BASE/models/sam2.1_hiera_b_plus/sam2.1_hiera_base_plus.pt" --sam-config "$REPO/third_party/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml" --reference-count 720 --fps 1 --iterations 30000 --colmap "$COL/usr/bin/colmap" --colmap-no-gpu)
brands=(riolight rio清爽 冰峰 凯撒 凯撒黑 可乐 哈尔滨 哈尔滨清爽 嘉士伯 无糖百事 科罗娜啤酒 红牛 酸梅汤)
run_one() {
  local brand="$1" gpu="$2" port="$3"
  echo "[queue] ${brand} gpu=${gpu} port=${port}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PY" build_reference_bank.py "${common[@]}" --brands "$brand" --gpu "$gpu" --port "$port" --reuse > "$MODEL/logs/${brand}.log" 2>&1
}
run_lane() {
  local gpu="$1" port="$2"
  shift 2
  local brand
  for brand in "$@"; do
    run_one "$brand" "$gpu" "$port"
    port=$((port + 1))
  done
}
lane1=()
lane3=()
index=0
for brand in "${brands[@]}"; do
  if (( index % 2 == 0 )); then
    lane1+=("$brand")
  else
    lane3+=("$brand")
  fi
  index=$((index + 1))
done
status=0
run_lane 1 6100 "${lane1[@]}" &
lane1_pid=$!
run_lane 3 6200 "${lane3[@]}" &
lane3_pid=$!
wait "$lane1_pid" || status=1
wait "$lane3_pid" || status=1
if [ "$status" -ne 0 ]; then
  exit "$status"
fi
"$PY" build_reference_bank.py "${common[@]}" --brands "${brands[@]}" --gpu 1 --port 6300 --reuse > "$MODEL/logs/aggregate.log" 2>&1
