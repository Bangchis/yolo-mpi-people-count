#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo_common.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
report_dir="${YOLO_REPORT_DIR:-results/report_mot17_mini_${timestamp}}"

export YOLO_SOURCE="${YOLO_SOURCE:-data/mot17-mini/MOT17-02-SDP-300_960x540.mp4}"
export YOLO_GT_COUNTS="${YOLO_GT_COUNTS:-data/mot17-mini/MOT17-02-SDP-300_counts.csv}"
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
export YOLO_SCHEDULE="${YOLO_SCHEDULE:-dynamic}"
export YOLO_MASTER_COMPUTE="${YOLO_MASTER_COMPUTE:-1}"
export YOLO_CHUNK_SIZE="${YOLO_CHUNK_SIZE:-1}"
export YOLO_RENDER_VIDEO="${YOLO_RENDER_VIDEO:-0}"
export YOLO_DETECTOR="${YOLO_DETECTOR:-yolo}"

if [[ "${YOLO_REPORT_QUICK:-0}" == "1" ]]; then
  correctness_frames="${YOLO_CORRECTNESS_FRAMES:-5}"
  accuracy_frames="${YOLO_ACCURACY_FRAMES:-10}"
  granularity_frames="${YOLO_GRANULARITY_FRAMES:-10}"
  export YOLO_FIND_FRAME_LIST="${YOLO_FIND_FRAME_LIST:-5 10}"
  export YOLO_GRANULARITY_GRIDS="${YOLO_GRANULARITY_GRIDS:-1x1 2x2}"
  export YOLO_P_LIST="${YOLO_P_LIST:-1 2}"
  export YOLO_SPEEDUP_FRAMES="${YOLO_SPEEDUP_FRAMES:-10}"
else
  correctness_frames="${YOLO_CORRECTNESS_FRAMES:-30}"
  accuracy_frames="${YOLO_ACCURACY_FRAMES:-300}"
  granularity_frames="${YOLO_GRANULARITY_FRAMES:-150}"
  export YOLO_FIND_FRAME_LIST="${YOLO_FIND_FRAME_LIST:-30 60 100 150}"
  export YOLO_GRANULARITY_GRIDS="${YOLO_GRANULARITY_GRIDS:-1x1 2x2 4x3}"
  export YOLO_P_LIST="${YOLO_P_LIST:-1 2 4 8 12}"
  export YOLO_SPEEDUP_FRAMES="${YOLO_SPEEDUP_FRAMES:-300}"
fi

report_np="${YOLO_REPORT_MPI_NP:-${YOLO_NP:-12}}"
report_hostfile="${YOLO_REPORT_HOSTFILE:-${YOLO_CORE_HOSTFILE:-configs/hosts_macos_core}}"
report_use_hostfile="${YOLO_USE_HOSTFILE:-1}"
main_tile_grid="${YOLO_TILE_GRID:-4x3}"

if [[ ! -f "$YOLO_SOURCE" ]]; then
  cat >&2 <<EOF
Missing YOLO_SOURCE=$YOLO_SOURCE
Download the MOT17-mini assets first, for example:
  $python_bin scripts/download_hf_assets.py \\
    --repo-id Bangchis/yolo-mpi-people-count-assets \\
    --asset data/mot17-mini/MOT17-02-SDP-300_960x540.mp4 \\
    --asset data/mot17-mini/MOT17-02-SDP-300_counts.csv
EOF
  exit 1
fi

if [[ ! -f "$YOLO_GT_COUNTS" ]]; then
  echo "Missing YOLO_GT_COUNTS=$YOLO_GT_COUNTS" >&2
  exit 1
fi

mkdir -p "$report_dir"/{dataset,correctness,accuracy,find_N,granularity,speedup}

echo "YOLO_REPORT_DIR=$report_dir"
echo "PHASE 0/6: build and runtime preparation"
bash scripts/build.sh
prepare_yolo_runtime

"$python_bin" scripts/probe_video.py --source "$YOLO_SOURCE" --format json > "$report_dir/dataset/video_info.json"
cp "$YOLO_GT_COUNTS" "$report_dir/dataset/gt_counts.csv"

git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
cat > "$report_dir/manifest.txt" <<EOF
run_dir=$report_dir
timestamp=$timestamp
git_rev=$git_rev
source=$YOLO_SOURCE
gt_counts=$YOLO_GT_COUNTS
model=$YOLO_MODEL
device=$YOLO_DEVICE
imgsz=$YOLO_IMGSZ
conf=$YOLO_CONF
iou=$YOLO_IOU
schedule=$YOLO_SCHEDULE
chunk_size=$YOLO_CHUNK_SIZE
master_compute=$YOLO_MASTER_COMPUTE
main_tile_grid=$main_tile_grid
report_np=$report_np
report_hostfile=$report_hostfile
use_hostfile=$report_use_hostfile
find_frame_list=$YOLO_FIND_FRAME_LIST
granularity_grids=$YOLO_GRANULARITY_GRIDS
speedup_p_list=$YOLO_P_LIST
speedup_frames=$YOLO_SPEEDUP_FRAMES
EOF

run_perf() {
  local out_dir="$1"
  local frames="$2"
  local np="$3"
  local use_hostfile="$4"
  local hostfile="$5"
  local tile_grid="$6"
  local run_label="$7"

  YOLO_RUN_DIR="$out_dir" \
  YOLO_PERF_FRAMES="$frames" \
  YOLO_NP="$np" \
  YOLO_USE_HOSTFILE="$use_hostfile" \
  YOLO_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_RENDER_VIDEO=0 \
  YOLO_PERF_SCHEDULE="$YOLO_SCHEDULE" \
  bash scripts/run_demo_perf.sh

  echo "${run_label}_DIR=$out_dir"
}

echo
echo "PHASE 1/6: correctness, serial vs MPI"
serial_dir="$report_dir/correctness/serial"
mpi_dir="$report_dir/correctness/mpi"
run_perf "$serial_dir" "$correctness_frames" 1 0 "$report_hostfile" "$main_tile_grid" "CORRECTNESS_SERIAL"
run_perf "$mpi_dir" "$correctness_frames" "$report_np" "$report_use_hostfile" "$report_hostfile" "$main_tile_grid" "CORRECTNESS_MPI"

correctness_status=0
"$python_bin" scripts/compare_frame_counts.py \
  --serial "$serial_dir/frame_counts.csv" \
  --mpi "$mpi_dir/frame_counts.csv" \
  --output "$report_dir/correctness/correctness_compare.csv" \
  --per-frame-output "$report_dir/correctness/correctness_per_frame.csv" || correctness_status=$?
if [[ "$correctness_status" != "0" ]]; then
  echo "WARNING: serial vs MPI correctness comparison reported mismatches; see correctness_compare.csv" >&2
fi

echo
echo "PHASE 2/6: YOLO count accuracy against MOT17 ground truth"
accuracy_pred_dir="$report_dir/accuracy/prediction"
run_perf "$accuracy_pred_dir" "$accuracy_frames" "$report_np" "$report_use_hostfile" "$report_hostfile" "$main_tile_grid" "ACCURACY_PREDICTION"
"$python_bin" scripts/evaluate_count_accuracy.py \
  --predicted "$accuracy_pred_dir/frame_counts.csv" \
  --ground-truth "$YOLO_GT_COUNTS" \
  --summary-output "$report_dir/accuracy/accuracy.csv" \
  --per-frame-output "$report_dir/accuracy/per_frame_accuracy.csv"
"$python_bin" plots/plot_count_error.py \
  --input "$report_dir/accuracy/per_frame_accuracy.csv" \
  --output "$report_dir/accuracy/count_error_plot.png"

echo
echo "PHASE 3/6: find N"
YOLO_RUN_DIR="$report_dir/find_N" \
YOLO_NP="$report_np" \
YOLO_USE_HOSTFILE="$report_use_hostfile" \
YOLO_HOSTFILE="$report_hostfile" \
YOLO_TILE_GRID="$main_tile_grid" \
bash scripts/run_find_N.sh
"$python_bin" plots/plot_find_n.py \
  --input "$report_dir/find_N/raw/find_N.csv" \
  --output "$report_dir/find_N/figures/find_N_runtime.png"

echo
echo "PHASE 4/6: granularity and load balance"
granularity_overview="$report_dir/granularity/granularity_overview.csv"
overview_initialized=0
for grid in $YOLO_GRANULARITY_GRIDS; do
  safe_grid="${grid//x/_x_}"
  grid_dir="$report_dir/granularity/grid_${grid}"
  run_perf "$grid_dir" "$granularity_frames" "$report_np" "$report_use_hostfile" "$report_hostfile" "$grid" "GRANULARITY_${safe_grid}"
  "$python_bin" plots/plot_rank_metrics.py \
    --input "$grid_dir/rank_metrics.csv" \
    --output "$grid_dir/rank_metrics_stacked.png" \
    --summary-output "$grid_dir/granularity_summary.csv" \
    --label "grid_${grid}"
  if [[ "$overview_initialized" == "0" ]]; then
    cat "$grid_dir/granularity_summary.csv" > "$granularity_overview"
    overview_initialized=1
  else
    tail -n +2 "$grid_dir/granularity_summary.csv" >> "$granularity_overview"
  fi
done

echo
echo "PHASE 5/6: speedup sweep"
YOLO_RUN_DIR="$report_dir/speedup" \
YOLO_USE_HOSTFILE="$report_use_hostfile" \
YOLO_SWEEP_HOSTFILE="$report_hostfile" \
YOLO_TILE_GRID="$main_tile_grid" \
bash scripts/run_speedup_sweep.sh

echo
echo "PHASE 6/6: done"
echo "YOLO_REPORT_DONE=YES"
echo "YOLO_REPORT_DIR=$report_dir"
echo "Correctness status: $correctness_status"
