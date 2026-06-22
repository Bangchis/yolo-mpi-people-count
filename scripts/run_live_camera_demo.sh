#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${YOLO_RUN_DIR:-results/live_camera_${timestamp}}"
mkdir -p "$run_dir"

np="${YOLO_NP:-3}"
hostfile="${YOLO_LIVE_HOSTFILE:-configs/hosts_macos_live}"
mpi=()
make_mpi_prefix "$np" "$hostfile" "${YOLO_USE_HOSTFILE:-1}" mpi

cpp_cmd=(build/yolo_mpi_cpp
  --live 1
  --source "camera:${YOLO_CAMERA_INDEX:-0}"
  --camera-index "${YOLO_CAMERA_INDEX:-0}"
  --live-video-source "${YOLO_LIVE_VIDEO_SOURCE:-}"
  --camera-script "${YOLO_CAMERA_SCRIPT:-scripts/camera_tile_source.py}"
  --viewer-script "${YOLO_VIEWER_SCRIPT:-scripts/live_viewer.py}"
  --live-view "${YOLO_LIVE_VIEW:-1}"
  --live-master-compute "${YOLO_LIVE_MASTER_COMPUTE:-1}"
  --live-anchor-full-frame "${YOLO_LIVE_ANCHOR_FULL_FRAME:-0}"
  --live-temporal-dedup "${YOLO_LIVE_TEMPORAL_DEDUP:-0}"
  --live-temporal-center "${YOLO_LIVE_TEMPORAL_CENTER:-0.55}"
  --live-temporal-ios "${YOLO_LIVE_TEMPORAL_IOS:-0.35}"
  --live-fps "${YOLO_LIVE_FPS:-0}"
  --jpeg-quality "${YOLO_JPEG_QUALITY:-80}"
  --model "${YOLO_MODEL:-models/yolo11n.pt}"
  --device "${YOLO_DEVICE:-mps}"
  --imgsz "${YOLO_LIVE_IMGSZ:-${YOLO_IMGSZ:-416}}"
  --conf "${YOLO_CONF:-0.35}"
  --iou "${YOLO_IOU:-0.50}"
  --tile-grid "${YOLO_LIVE_TILE_GRID:-${YOLO_TILE_GRID:-3x1}}"
  --overlap "${YOLO_LIVE_OVERLAP:-${YOLO_OVERLAP:-64}}"
  --tile-owner-filter "${YOLO_TILE_OWNER_FILTER:-1}"
  --dedup-ios "${YOLO_DEDUP_IOS:-0.70}"
  --dedup-center "${YOLO_DEDUP_CENTER:-0.30}"
  --dedup-axis-overlap "${YOLO_DEDUP_AXIS_OVERLAP:-0.70}"
  --dedup-gap "${YOLO_DEDUP_GAP:-0.08}"
  --dedup-near-camera "${YOLO_DEDUP_NEAR_CAMERA:-0}"
  --dedup-large-area-ratio "${YOLO_DEDUP_LARGE_AREA_RATIO:-0.12}"
  --dedup-merge "${YOLO_DEDUP_MERGE:-1}"
  --schedule dynamic
  --chunk-size "${YOLO_CHUNK_SIZE:-1}"
  --frames "${YOLO_LIVE_FRAMES:-100}"
  --width "${YOLO_FRAME_WIDTH:-1280}"
  --height "${YOLO_FRAME_HEIGHT:-720}"
  --detector "${YOLO_DETECTOR:-yolo}"
  --python "$python_bin"
  --worker-script "${YOLO_WORKER_SCRIPT:-scripts/yolo_worker.py}"
  --cpu-fallback "${YOLO_CPU_FALLBACK:-1}"
  --sleep-ms "${YOLO_SLEEP_MS:-0}"
  --verify 0
  --write-video 0
  --run-id "live_camera_${timestamp}"
  --output "$run_dir")

rank_cmd=()
make_rank_command rank_cmd "${cpp_cmd[@]}"

echo "COMMAND: ${mpi[*]} ${rank_cmd[*]}"
"${mpi[@]}" "${rank_cmd[@]}" | tee "$run_dir/live_camera.log"
pull_rank_output_if_needed "$run_dir"

echo "LIVE_CAMERA_DIR=$run_dir"
