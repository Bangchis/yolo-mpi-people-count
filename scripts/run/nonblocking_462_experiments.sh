#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${YOLO_NB_RUN_DIR:-results/nonblocking_462_${timestamp}}"
hostfile="${YOLO_NB_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}"
np="${YOLO_NB_NP:-12}"

mini_source="${YOLO_NB_MINI_SOURCE:-data/mot17-mini/MOT17-02-SDP-300_960x540.mp4}"
mini_gt="${YOLO_NB_MINI_GT:-data/mot17-mini/MOT17-02-SDP-300_counts.csv}"
find_source="${YOLO_NB_FIND_SOURCE:-data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4}"
granularity_source="${YOLO_NB_GRANULARITY_SOURCE:-data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4}"
speedup_source="${YOLO_NB_SPEEDUP_SOURCE:-data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4}"

if [[ "${YOLO_NB_QUICK:-0}" == "1" ]]; then
  correctness_frames="${YOLO_NB_CORRECTNESS_FRAMES:-5}"
  accuracy_frames="${YOLO_NB_ACCURACY_FRAMES:-10}"
  find_frame_list="${YOLO_NB_FIND_FRAME_LIST:-10 20}"
  granularity_frames="${YOLO_NB_GRANULARITY_FRAMES:-10}"
  granularity_grids="${YOLO_NB_GRANULARITY_GRIDS:-2x2 4x3}"
  speedup_frames="${YOLO_NB_SPEEDUP_FRAMES:-20}"
  p_list="${YOLO_NB_P_LIST:-1 2 4}"
else
  correctness_frames="${YOLO_NB_CORRECTNESS_FRAMES:-30}"
  accuracy_frames="${YOLO_NB_ACCURACY_FRAMES:-300}"
  find_frame_list="${YOLO_NB_FIND_FRAME_LIST:-300 600 837}"
  granularity_frames="${YOLO_NB_GRANULARITY_FRAMES:-600}"
  granularity_grids="${YOLO_NB_GRANULARITY_GRIDS:-2x2 4x3 5x4}"
  speedup_frames="${YOLO_NB_SPEEDUP_FRAMES:-1200}"
  p_list="${YOLO_NB_P_LIST:-1 2 4 8 12}"
fi

tile_grid="${YOLO_NB_TILE_GRID:-5x4}"
modes="${YOLO_NB_COMM_MODES:-blocking nonblocking streaming}"

export YOLO_MODEL="${YOLO_MODEL:-models/yolo11n.pt}"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-320}"
export YOLO_CONF="${YOLO_CONF:-0.35}"
export YOLO_IOU="${YOLO_IOU:-0.50}"
export YOLO_OVERLAP="${YOLO_OVERLAP:-64}"
export YOLO_TILE_OWNER_FILTER="${YOLO_TILE_OWNER_FILTER:-1}"
export YOLO_DEDUP_IOS="${YOLO_DEDUP_IOS:-0.70}"
export YOLO_DEDUP_CENTER="${YOLO_DEDUP_CENTER:-0.30}"
export YOLO_DEDUP_AXIS_OVERLAP="${YOLO_DEDUP_AXIS_OVERLAP:-0.70}"
export YOLO_DEDUP_GAP="${YOLO_DEDUP_GAP:-0.08}"
export YOLO_DEDUP_NEAR_CAMERA="${YOLO_DEDUP_NEAR_CAMERA:-0}"
export YOLO_DEDUP_LARGE_AREA_RATIO="${YOLO_DEDUP_LARGE_AREA_RATIO:-0.12}"
export YOLO_DEDUP_MERGE="${YOLO_DEDUP_MERGE:-1}"
export YOLO_SCHEDULE=static
export YOLO_MASTER_COMPUTE="${YOLO_MASTER_COMPUTE:-1}"
export YOLO_CHUNK_SIZE="${YOLO_CHUNK_SIZE:-1}"
export YOLO_STREAM_BATCH_TASKS="${YOLO_STREAM_BATCH_TASKS:-20}"
export YOLO_STREAM_MAX_PENDING="${YOLO_STREAM_MAX_PENDING:-2}"
export YOLO_DETECTOR="${YOLO_DETECTOR:-yolo}"
export YOLO_USE_HOSTFILE="${YOLO_USE_HOSTFILE:-1}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_perf_case() {
  local out="$1"
  local source="$2"
  local frames="$3"
  local ranks="$4"
  local use_hostfile="$5"
  local grid="$6"
  local comm_mode="$7"

  YOLO_SOURCE="$source" \
  YOLO_RUN_DIR="$out" \
  YOLO_PERF_FRAMES="$frames" \
  YOLO_NP="$ranks" \
  YOLO_USE_HOSTFILE="$use_hostfile" \
  YOLO_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$grid" \
  YOLO_PERF_SCHEDULE=static \
  YOLO_COMM_MODE="$comm_mode" \
  YOLO_RENDER_VIDEO=0 \
  bash scripts/run/demo_perf.sh
}

summarize_mode() {
  local mode_dir="$1"
  local mode="$2"
  local output="$3"

  python3 - "$mode_dir" "$mode" "$output" <<'PY'
import csv
import sys
from pathlib import Path

mode_dir = Path(sys.argv[1])
mode = sys.argv[2]
output = Path(sys.argv[3])

def read_one(path):
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))

def seconds(ms):
    return float(ms) / 1000.0

find_rows = list(csv.DictReader((mode_dir / "find_N" / "raw" / "find_N.csv").open(newline="", encoding="utf-8")))
find_600 = next((row for row in find_rows if row["frames"] == "600"), find_rows[-1])
grid_dir = mode_dir / "granularity" / "grid_5x4"
if not grid_dir.exists():
    grid_dirs = sorted((mode_dir / "granularity").glob("grid_*"))
    grid_dir = grid_dirs[-1]
granularity_summary = read_one(grid_dir / "summary.csv")
speed_rows = list(csv.DictReader((mode_dir / "speedup" / "raw" / "speedup.csv").open(newline="", encoding="utf-8")))
speed_p12 = next((row for row in speed_rows if row["world_size"] == "12"), speed_rows[-1])
correctness = read_one(mode_dir / "correctness" / "correctness_compare.csv")
accuracy = read_one(mode_dir / "accuracy" / "accuracy.csv")

row = {
    "comm_mode": mode,
    "correctness_pass": correctness["correctness_pass"],
    "accuracy_mae": accuracy["mae"],
    "find_n_600_s": f"{seconds(find_600['total_ms_with_comm']):.6f}",
    "find_n_600_without_comm_s": f"{seconds(find_600['total_ms_without_comm']):.6f}",
    "granularity_5x4_s": f"{float(granularity_summary['total_ms_with_comm']) / 1000.0:.6f}",
    "speedup_p12_s": f"{float(speed_p12['total_ms_with_comm']) / 1000.0:.6f}",
    "speedup_p12": speed_p12["speedup_with_comm"],
    "efficiency_p12": speed_p12["efficiency_with_comm"],
}

exists = output.exists()
with output.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if not exists:
        writer.writeheader()
    writer.writerow(row)
PY
}

require_file "$mini_source"
require_file "$mini_gt"
require_file "$find_source"
require_file "$granularity_source"
require_file "$speedup_source"
require_file "$hostfile"

mkdir -p "$run_dir"

cat > "$run_dir/manifest.txt" <<EOF
run_dir=$run_dir
timestamp=$timestamp
hostfile=$hostfile
placement=master:4,node1:6,node2:2
np=$np
modes=$modes
schedule=static
stream_batch_tasks=$YOLO_STREAM_BATCH_TASKS
stream_max_pending=$YOLO_STREAM_MAX_PENDING
device=$YOLO_DEVICE
imgsz=$YOLO_IMGSZ
mini_source=$mini_source
find_source=$find_source
granularity_source=$granularity_source
speedup_source=$speedup_source
correctness_frames=$correctness_frames
accuracy_frames=$accuracy_frames
find_frame_list=$find_frame_list
granularity_frames=$granularity_frames
granularity_grids=$granularity_grids
speedup_frames=$speedup_frames
p_list=$p_list
EOF

echo "NONBLOCKING_462_RUN_DIR=$run_dir"
echo "PHASE 0: build and runtime preparation"
bash scripts/build.sh
prepare_yolo_runtime

overview="$run_dir/comm_mode_overview.csv"
rm -f "$overview"

for mode in $modes; do
  mode_dir="$run_dir/$mode"
  mkdir -p "$mode_dir"/{correctness,accuracy,find_N,granularity,speedup,figures}

  echo
  echo "MODE=$mode PHASE 1: correctness"
  serial_dir="$mode_dir/correctness/serial"
  mpi_dir="$mode_dir/correctness/mpi"
  run_perf_case "$serial_dir" "$mini_source" "$correctness_frames" 1 0 "$tile_grid" "$mode"
  run_perf_case "$mpi_dir" "$mini_source" "$correctness_frames" "$np" 1 "$tile_grid" "$mode"
  "$python_bin" scripts/report/compare_frame_counts.py \
    --serial "$serial_dir/frame_counts.csv" \
    --mpi "$mpi_dir/frame_counts.csv" \
    --output "$mode_dir/correctness/correctness_compare.csv" \
    --per-frame-output "$mode_dir/correctness/correctness_per_frame.csv"

  echo "MODE=$mode PHASE 2: accuracy"
  accuracy_pred="$mode_dir/accuracy/prediction"
  run_perf_case "$accuracy_pred" "$mini_source" "$accuracy_frames" "$np" 1 "$tile_grid" "$mode"
  "$python_bin" scripts/report/evaluate_count_accuracy.py \
    --predicted "$accuracy_pred/frame_counts.csv" \
    --ground-truth "$mini_gt" \
    --summary-output "$mode_dir/accuracy/accuracy.csv" \
    --per-frame-output "$mode_dir/accuracy/per_frame_accuracy.csv"
  "$python_bin" scripts/report/plots/plot_count_error.py \
    --input "$mode_dir/accuracy/per_frame_accuracy.csv" \
    --output "$mode_dir/accuracy/count_error_plot.png"

  echo "MODE=$mode PHASE 3: find N"
  YOLO_SOURCE="$find_source" \
  YOLO_RUN_DIR="$mode_dir/find_N" \
  YOLO_NP="$np" \
  YOLO_USE_HOSTFILE=1 \
  YOLO_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_FIND_FRAME_LIST="$find_frame_list" \
  YOLO_COMM_MODE="$mode" \
  bash scripts/run/find_N.sh
  "$python_bin" scripts/report/plots/plot_find_n.py \
    --input "$mode_dir/find_N/raw/find_N.csv" \
    --output "$mode_dir/find_N/figures/find_N_runtime.png"

  echo "MODE=$mode PHASE 4: granularity"
  granularity_overview="$mode_dir/granularity/granularity_overview.csv"
  overview_initialized=0
  for grid in $granularity_grids; do
    grid_dir="$mode_dir/granularity/grid_${grid}"
    run_perf_case "$grid_dir" "$granularity_source" "$granularity_frames" "$np" 1 "$grid" "$mode"
    "$python_bin" scripts/report/plots/plot_rank_metrics.py \
      --input "$grid_dir/rank_metrics.csv" \
      --output "$grid_dir/rank_metrics_stacked.png" \
      --summary-output "$grid_dir/granularity_summary.csv" \
      --label "${mode}_grid_${grid}"
    if [[ "$overview_initialized" == "0" ]]; then
      cat "$grid_dir/granularity_summary.csv" > "$granularity_overview"
      overview_initialized=1
    else
      tail -n +2 "$grid_dir/granularity_summary.csv" >> "$granularity_overview"
    fi
  done
  "$python_bin" scripts/report/plots/plot_granularity_overview.py \
    --input "$granularity_overview" \
    --output "$mode_dir/granularity/granularity_overview.png"

  echo "MODE=$mode PHASE 5: speedup"
  YOLO_SOURCE="$speedup_source" \
  YOLO_RUN_DIR="$mode_dir/speedup" \
  YOLO_USE_HOSTFILE=1 \
  YOLO_SWEEP_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_SPEEDUP_FRAMES="$speedup_frames" \
  YOLO_P_LIST="$p_list" \
  YOLO_COMM_MODE="$mode" \
  bash scripts/run/speedup_sweep.sh

  summarize_mode "$mode_dir" "$mode" "$overview"
done

"$python_bin" scripts/report/plots/plot_comm_mode_comparison.py \
  --input "$overview" \
  --output "$run_dir/figures/comm_mode_comparison.png"

echo
echo "NONBLOCKING_462_DONE=YES"
echo "NONBLOCKING_462_RUN_DIR=$run_dir"
echo "NONBLOCKING_462_OVERVIEW=$overview"
