#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

timestamp="$(date +%Y%m%d-%H%M%S)"
suite_dir="${YOLO_SUITE_DIR:-results/mot17_fullseq_accuracy_suite_${timestamp}}"
mkdir -p "$suite_dir"

export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-320}"
export YOLO_TILE_GRID="${YOLO_TILE_GRID:-4x3}"
export YOLO_PERF_SCHEDULE=static
export YOLO_MASTER_COMPUTE="${YOLO_MASTER_COMPUTE:-1}"
export YOLO_RENDER_VIDEO="${YOLO_RENDER_VIDEO:-0}"
export YOLO_NP="${YOLO_NP:-12}"
export YOLO_HOSTFILE="${YOLO_HOSTFILE:-configs/hosts_macos_core}"

sequence_specs="${YOLO_MOT17_SEQUENCE_SPECS:-\
MOT17-02-SDP:600 \
MOT17-05-SDP:837 \
MOT17-09-SDP:525 \
MOT17-10-SDP:654}"

cat > "$suite_dir/manifest.txt" <<EOF
created_at=$timestamp
purpose=MOT17 full-sequence YOLO-vs-ground-truth count accuracy suite
sequence_specs=$sequence_specs
device=$YOLO_DEVICE
imgsz=$YOLO_IMGSZ
tile_grid=$YOLO_TILE_GRID
schedule=$YOLO_PERF_SCHEDULE
master_compute=$YOLO_MASTER_COMPUTE
render_video=$YOLO_RENDER_VIDEO
detector=${YOLO_DETECTOR:-yolo}
np=${YOLO_NP:-3}
hostfile=$YOLO_HOSTFILE
use_hostfile=${YOLO_USE_HOSTFILE:-1}
EOF

overview="$suite_dir/accuracy_overview.csv"
echo "sequence,frames_compared,mae,rmse,mean_abs_percentage_error,exact_match_rate,mean_gt_count,mean_pred_count,run_dir" > "$overview"

for spec in $sequence_specs; do
  sequence="${spec%%:*}"
  frames="${spec##*:}"
  source="data/mot17-fullseq/${sequence}-${frames}_960x540.mp4"
  gt="data/mot17-fullseq/${sequence}-${frames}_counts.csv"
  out="$suite_dir/$sequence"

  if [[ ! -f "$source" || ! -f "$gt" ]]; then
    echo "Missing assets for $sequence. Expected $source and $gt" >&2
    exit 1
  fi

  echo "RUN_ACCURACY_SEQUENCE=$sequence FRAMES=$frames"
  YOLO_RUN_DIR="$out/prediction" \
  YOLO_SOURCE="$source" \
  YOLO_PERF_FRAMES="$frames" \
  bash scripts/run/demo_perf.sh

  "$python_bin" scripts/report/evaluate_count_accuracy.py \
    --predicted "$out/prediction/frame_counts.csv" \
    --ground-truth "$gt" \
    --summary-output "$out/accuracy.csv" \
    --per-frame-output "$out/per_frame_accuracy.csv"

  "$python_bin" scripts/report/plots/plot_count_error.py \
    --input "$out/per_frame_accuracy.csv" \
    --output "$out/count_error_plot.png"

  "$python_bin" - "$sequence" "$out/accuracy.csv" "$out/prediction" "$overview" <<'PY'
import csv
import sys

sequence, accuracy_csv, run_dir, overview = sys.argv[1:]
with open(accuracy_csv, newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
with open(overview, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        sequence,
        row["frames_compared"],
        row["mae"],
        row["rmse"],
        row["mean_abs_percentage_error"],
        row["exact_match_rate"],
        row["mean_gt_count"],
        row["mean_pred_count"],
        run_dir,
    ])
PY
done

echo "MOT17_FULLSEQ_ACCURACY_SUITE_DONE=YES"
echo "MOT17_FULLSEQ_ACCURACY_SUITE_DIR=$suite_dir"
