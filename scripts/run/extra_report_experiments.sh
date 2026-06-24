#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${YOLO_EXTRA_RUN_DIR:-results/extra_report_full_${timestamp}}"
mkdir -p "$run_dir"

source_video="${YOLO_EXTRA_SOURCE:-data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4}"
speedup_video="${YOLO_EXTRA_SPEEDUP_SOURCE:-data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4}"
model="${YOLO_MODEL:-models/yolo11n.pt}"
hostfile="${YOLO_EXTRA_HOSTFILE:-configs/hosts_macos_core}"
np="${YOLO_EXTRA_NP:-12}"
frames="${YOLO_EXTRA_FRAMES:-600}"
speedup_frames="${YOLO_EXTRA_SPEEDUP_FRAMES:-1200}"
tile_grid="${YOLO_EXTRA_TILE_GRID:-5x4}"
granularity_grids="${YOLO_EXTRA_GRIDS:-2x2 4x3 5x4}"
local_p_list="${YOLO_EXTRA_LOCAL_P_LIST:-1 2 4 8}"
cluster_p_list="${YOLO_EXTRA_CLUSTER_P_LIST:-1 2 4 8 12}"
speedup_repeats="${YOLO_EXTRA_SPEEDUP_REPEATS:-2}"

export YOLO_SOURCE="$source_video"
export YOLO_MODEL="$model"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
export YOLO_IMGSZ="${YOLO_IMGSZ:-320}"
export YOLO_CONF="${YOLO_CONF:-0.35}"
export YOLO_IOU="${YOLO_IOU:-0.50}"
export YOLO_TILE_OWNER_FILTER="${YOLO_TILE_OWNER_FILTER:-1}"
export YOLO_DEDUP_MERGE="${YOLO_DEDUP_MERGE:-1}"
export YOLO_RENDER_VIDEO=0
export YOLO_MASTER_COMPUTE="${YOLO_MASTER_COMPUTE:-1}"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_perf_case() {
  local out="$1"
  local case_frames="$2"
  local case_np="$3"
  local use_hostfile="$4"
  local case_hostfile="$5"
  local case_grid="$6"
  local schedule="$7"

  YOLO_RUN_DIR="$out" \
  YOLO_PERF_FRAMES="$case_frames" \
  YOLO_NP="$case_np" \
  YOLO_USE_HOSTFILE="$use_hostfile" \
  YOLO_HOSTFILE="$case_hostfile" \
  YOLO_TILE_GRID="$case_grid" \
  YOLO_PERF_SCHEDULE="$schedule" \
  bash scripts/run/demo_perf.sh
}

append_speedup_rows() {
  local input_csv="$1"
  local output_csv="$2"
  local label="$3"
  local repeat="$4"
  python3 - "$input_csv" "$output_csv" "$label" "$repeat" <<'PY'
import csv
import sys
from pathlib import Path

input_csv = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
label = sys.argv[3]
repeat = sys.argv[4]

exists = output_csv.exists()
with input_csv.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = ["label", "repeat"] + list(rows[0].keys())
with output_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    for row in rows:
        row = {"label": label, "repeat": repeat, **row}
        writer.writerow(row)
PY
}

summarize_speedup_repeats() {
  local input_csv="$1"
  local output_csv="$2"
  python3 - "$input_csv" "$output_csv" <<'PY'
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

input_csv = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
groups = defaultdict(list)

with input_csv.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[(row["label"], row["world_size"])].append(row)

fieldnames = [
    "label",
    "world_size",
    "repeats",
    "mean_total_ms_with_comm",
    "stdev_total_ms_with_comm",
    "mean_speedup_with_comm",
    "mean_efficiency_with_comm",
    "tile_grid",
    "schedule",
]
with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for (label, world_size), rows in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1]))):
        times = [float(r["total_ms_with_comm"]) for r in rows]
        speedups = [float(r["speedup_with_comm"]) for r in rows]
        effs = [float(r["efficiency_with_comm"]) for r in rows]
        writer.writerow({
            "label": label,
            "world_size": world_size,
            "repeats": len(rows),
            "mean_total_ms_with_comm": statistics.mean(times),
            "stdev_total_ms_with_comm": statistics.stdev(times) if len(times) > 1 else 0.0,
            "mean_speedup_with_comm": statistics.mean(speedups),
            "mean_efficiency_with_comm": statistics.mean(effs),
            "tile_grid": rows[0]["tile_grid"],
            "schedule": rows[0]["schedule"],
        })
PY
}

require_file "$source_video"
require_file "$speedup_video"
require_file "$model"
require_file "$hostfile"

cat > "$run_dir/manifest.txt" <<EOF
run_dir=$run_dir
source_video=$source_video
speedup_video=$speedup_video
model=$model
device=$YOLO_DEVICE
imgsz=$YOLO_IMGSZ
frames=$frames
speedup_frames=$speedup_frames
tile_grid=$tile_grid
granularity_grids=$granularity_grids
hostfile=$hostfile
np=$np
local_p_list=$local_p_list
cluster_p_list=$cluster_p_list
speedup_repeats=$speedup_repeats
EOF

echo "PHASE 0: full-cluster preflight"
YOLO_HOSTFILE="$hostfile" YOLO_EVIDENCE_NP=3 bash scripts/cluster/check_macos.sh

echo "PHASE 1: sync repository and assets to nodes"
bash scripts/cluster/sync_to_nodes.sh

echo "PHASE 2: local-only speedup on master, N=$frames"
YOLO_RUN_DIR="$run_dir/local_only_N${frames}" \
YOLO_USE_HOSTFILE=0 \
YOLO_P_LIST="$local_p_list" \
YOLO_SPEEDUP_FRAMES="$frames" \
YOLO_TILE_GRID="$tile_grid" \
YOLO_SCHEDULE=static \
bash scripts/run/speedup_sweep.sh

echo "PHASE 3: three-machine cluster speedup on N=$frames"
YOLO_RUN_DIR="$run_dir/cluster_N${frames}" \
YOLO_USE_HOSTFILE=1 \
YOLO_SWEEP_HOSTFILE="$hostfile" \
YOLO_P_LIST="$cluster_p_list" \
YOLO_SPEEDUP_FRAMES="$frames" \
YOLO_TILE_GRID="$tile_grid" \
YOLO_SCHEDULE=static \
bash scripts/run/speedup_sweep.sh

python3 - "$run_dir/local_only_N${frames}/raw/speedup.csv" "$run_dir/cluster_N${frames}/raw/speedup.csv" "$run_dir/local_vs_cluster.csv" <<'PY'
import csv
import sys
from pathlib import Path

local_csv = Path(sys.argv[1])
cluster_csv = Path(sys.argv[2])
out_csv = Path(sys.argv[3])

rows = []
for label, path in [("local_only", local_csv), ("cluster_3_machine", cluster_csv)]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({"label": label, **row})

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["label"] + list(rows[0].keys())[1:])
    writer.writeheader()
    writer.writerows(rows)
PY

echo "PHASE 4: granularity on N=$frames"
gran_dir="$run_dir/granularity_N${frames}"
mkdir -p "$gran_dir"
overview="$gran_dir/granularity_overview.csv"
overview_initialized=0
for grid in $granularity_grids; do
  out="$gran_dir/grid_${grid}"
  run_perf_case "$out" "$frames" "$np" 1 "$hostfile" "$grid" static
  "$python_bin" scripts/report/plots/plot_rank_metrics.py \
    --input "$out/rank_metrics.csv" \
    --output "$out/rank_metrics_stacked.png" \
    --summary-output "$out/granularity_summary.csv" \
    --label "grid_${grid}"
  if [[ "$overview_initialized" == "0" ]]; then
    cat "$out/granularity_summary.csv" > "$overview"
    overview_initialized=1
  else
    tail -n +2 "$out/granularity_summary.csv" >> "$overview"
  fi
done
"$python_bin" scripts/report/plots/plot_granularity_overview.py \
  --input "$overview" \
  --output "$gran_dir/granularity_overview.png"

echo "PHASE 5: repeated speedup on 2N=$speedup_frames"
repeat_raw="$run_dir/speedup_repeats_raw.csv"
for repeat in $(seq 1 "$speedup_repeats"); do
  repeat_dir="$run_dir/speedup_2N_repeat_${repeat}"
  YOLO_SOURCE="$speedup_video" \
  YOLO_RUN_DIR="$repeat_dir" \
  YOLO_USE_HOSTFILE=1 \
  YOLO_SWEEP_HOSTFILE="$hostfile" \
  YOLO_P_LIST="$cluster_p_list" \
  YOLO_SPEEDUP_FRAMES="$speedup_frames" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_SCHEDULE=static \
  bash scripts/run/speedup_sweep.sh
  append_speedup_rows "$repeat_dir/raw/speedup.csv" "$repeat_raw" "cluster_2N" "$repeat"
done
summarize_speedup_repeats "$repeat_raw" "$run_dir/speedup_repeats_summary.csv"

echo "EXTRA_REPORT_FULL_DONE=YES"
echo "EXTRA_REPORT_FULL_DIR=$run_dir"
