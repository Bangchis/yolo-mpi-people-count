#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${YOLO_RUN_DIR:-results/demo_perf_${timestamp}}"
mkdir -p "$run_dir"

np="${YOLO_NP:-3}"
hostfile="${YOLO_HOSTFILE:-configs/hosts_macos_gpu}"
mpi=()
make_mpi_prefix "$np" "$hostfile" "${YOLO_USE_HOSTFILE:-1}" mpi

cpp_cmd=(build/yolo_mpi_cpp
  --source "${YOLO_SOURCE:-data/classroom.mp4}"
  --model "${YOLO_MODEL:-yolo11n.pt}"
  --device "${YOLO_DEVICE:-mps}"
  --imgsz "${YOLO_IMGSZ:-512}"
  --conf "${YOLO_CONF:-0.35}"
  --iou "${YOLO_IOU:-0.50}"
  --tile-grid "${YOLO_TILE_GRID:-1x1}"
  --overlap "${YOLO_OVERLAP:-64}"
  --tile-owner-filter "${YOLO_TILE_OWNER_FILTER:-1}"
  --dedup-ios "${YOLO_DEDUP_IOS:-0.70}"
  --dedup-center "${YOLO_DEDUP_CENTER:-0.30}"
  --dedup-axis-overlap "${YOLO_DEDUP_AXIS_OVERLAP:-0.70}"
  --dedup-gap "${YOLO_DEDUP_GAP:-0.08}"
  --dedup-near-camera "${YOLO_DEDUP_NEAR_CAMERA:-0}"
  --dedup-large-area-ratio "${YOLO_DEDUP_LARGE_AREA_RATIO:-0.12}"
  --dedup-merge "${YOLO_DEDUP_MERGE:-1}"
  --schedule "${YOLO_PERF_SCHEDULE:-static}"
  --comm-mode "${YOLO_COMM_MODE:-blocking}"
  --chunk-size "${YOLO_CHUNK_SIZE:-1}"
  --frames "${YOLO_PERF_FRAMES:-100}"
  --width "${YOLO_FRAME_WIDTH:-1280}"
  --height "${YOLO_FRAME_HEIGHT:-720}"
  --detector "${YOLO_DETECTOR:-yolo}"
  --python "$python_bin"
  --worker-script "${YOLO_WORKER_SCRIPT:-scripts/runtime/yolo_worker.py}"
  --cpu-fallback "${YOLO_CPU_FALLBACK:-1}"
  --sleep-ms "${YOLO_SLEEP_MS:-0}"
  --master-compute "${YOLO_MASTER_COMPUTE:-1}"
  --verify 0
  --run-id "demo_perf_${timestamp}"
  --output "$run_dir")
if [[ -n "${YOLO_DETECTOR_COMMAND:-}" ]]; then
  cpp_cmd+=(--detector-command "$YOLO_DETECTOR_COMMAND")
fi
rank_cmd=()
make_rank_command rank_cmd "${cpp_cmd[@]}"

echo "COMMAND: ${mpi[*]} ${rank_cmd[*]}"
"${mpi[@]}" "${rank_cmd[@]}" | tee "$run_dir/demo_perf.log"
pull_rank_output_if_needed "$run_dir"

if [[ "${YOLO_RENDER_VIDEO:-1}" == "1" && -f "${YOLO_SOURCE:-data/classroom.mp4}" ]]; then
  "$python_bin" scripts/runtime/render_demo_video.py \
    --source "${YOLO_SOURCE:-data/classroom.mp4}" \
    --bboxes "$run_dir/bboxes.csv" \
    --frames "${YOLO_PERF_FRAMES:-100}" \
    --output "$run_dir/demo/people_count_output.mp4" || true
fi

echo "DEMO_PERF_DIR=$run_dir"
