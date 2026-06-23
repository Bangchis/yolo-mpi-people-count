#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

run_dir="${YOLO_RUN_DIR:-results/scheduler_comparison_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_dir"/{static,dynamic,figures}

frames="${YOLO_SCHED_COMPARE_FRAMES:-${YOLO_PERF_FRAMES:-150}}"
tile_grid="${YOLO_SCHED_COMPARE_TILE_GRID:-${YOLO_TILE_GRID:-4x3}}"
np="${YOLO_SCHED_COMPARE_NP:-${YOLO_NP:-12}}"
hostfile="${YOLO_SCHED_COMPARE_HOSTFILE:-${YOLO_HOSTFILE:-${YOLO_CORE_HOSTFILE:-configs/hosts_macos_core}}}"
use_hostfile="${YOLO_USE_HOSTFILE:-1}"

run_case() {
  local schedule="$1"
  local out="$run_dir/$schedule"

  YOLO_RUN_DIR="$out" \
  YOLO_PERF_FRAMES="$frames" \
  YOLO_NP="$np" \
  YOLO_USE_HOSTFILE="$use_hostfile" \
  YOLO_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_RENDER_VIDEO=0 \
  YOLO_PERF_SCHEDULE="$schedule" \
  bash scripts/run/demo_perf.sh

  "$python_bin" scripts/report/plots/plot_rank_metrics.py \
    --input "$out/rank_metrics.csv" \
    --output "$out/rank_metrics_stacked.png" \
    --summary-output "$out/rank_summary.csv" \
    --label "$schedule"
}

cat > "$run_dir/manifest.txt" <<EOF
run_dir=$run_dir
frames=$frames
tile_grid=$tile_grid
np=$np
hostfile=$hostfile
use_hostfile=$use_hostfile
device=${YOLO_DEVICE:-cpu}
imgsz=${YOLO_IMGSZ:-320}
purpose=Compare static block-cyclic scheduling against dynamic master-worker scheduling.
EOF

run_case static
run_case dynamic

comparison_csv="$run_dir/scheduler_comparison.csv"
python3 - "$run_dir" "$comparison_csv" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])

rows = []
for schedule in ["static", "dynamic"]:
    summary_path = run_dir / schedule / "summary.csv"
    rank_summary_path = run_dir / schedule / "rank_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary = next(csv.DictReader(f))
    with rank_summary_path.open(newline="", encoding="utf-8") as f:
        rank_summary = next(csv.DictReader(f))

    rows.append({
        "schedule": schedule,
        "world_size": summary["world_size"],
        "frames": summary["frames"],
        "tile_grid": summary["tile_grid"],
        "num_tasks": summary["num_tasks"],
        "total_ms_with_comm": summary["total_ms_with_comm"],
        "total_ms_without_comm": summary["total_ms_without_comm"],
        "load_imbalance": summary["load_imbalance"],
        "comm_s_total": rank_summary["comm_s_total"],
        "idle_s_total": rank_summary["idle_s_total"],
        "idle_gap_ratio": rank_summary["idle_gap_ratio"],
        "load_balance_pass": rank_summary["load_balance_pass"],
    })

with output.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "schedule",
        "world_size",
        "frames",
        "tile_grid",
        "num_tasks",
        "total_ms_with_comm",
        "total_ms_without_comm",
        "load_imbalance",
        "comm_s_total",
        "idle_s_total",
        "idle_gap_ratio",
        "load_balance_pass",
    ])
    writer.writeheader()
    writer.writerows(rows)
PY

"$python_bin" scripts/report/plots/plot_scheduler_comparison.py \
  --input "$comparison_csv" \
  --output "$run_dir/figures/scheduler_comparison.png"

echo "SCHEDULER_COMPARISON_DONE=YES"
echo "SCHEDULER_COMPARISON_DIR=$run_dir"
echo "SCHEDULER_COMPARISON_CSV=$comparison_csv"
