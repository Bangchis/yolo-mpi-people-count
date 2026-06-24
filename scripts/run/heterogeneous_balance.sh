#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

run_dir="${YOLO_RUN_DIR:-results/heterogeneous_balance_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_dir"/{uniform_24,weighted_24}

frames="${YOLO_HET_FRAMES:-${YOLO_PERF_FRAMES:-120}}"
tile_grid="${YOLO_HET_TILE_GRID:-${YOLO_TILE_GRID:-5x4}}"
np="${YOLO_HET_NP:-24}"

run_case() {
  local label="$1"
  local hostfile="$2"
  local out="$run_dir/$label"

  YOLO_RUN_DIR="$out" \
  YOLO_PERF_FRAMES="$frames" \
  YOLO_NP="$np" \
  YOLO_USE_HOSTFILE=1 \
  YOLO_HOSTFILE="$hostfile" \
  YOLO_TILE_GRID="$tile_grid" \
  YOLO_RENDER_VIDEO=0 \
  YOLO_PERF_SCHEDULE=static \
  bash scripts/run/demo_perf.sh

  "$python_bin" scripts/report/plots/plot_rank_metrics.py \
    --input "$out/rank_metrics.csv" \
    --output "$out/rank_metrics_stacked.png" \
    --summary-output "$out/granularity_summary.csv" \
    --label "$label"

  "$python_bin" scripts/report/summarize_host_metrics.py \
    --input "$out/rank_metrics.csv" \
    --output "$out/host_metrics.csv" \
    --label "$label"
}

cat > "$run_dir/manifest.txt" <<EOF
run_dir=$run_dir
frames=$frames
tile_grid=$tile_grid
np=$np
schedule=static
device=${YOLO_DEVICE:-cpu}
imgsz=${YOLO_IMGSZ:-320}
uniform_hostfile=${YOLO_HET_UNIFORM_HOSTFILE:-configs/hosts_macos_cpu_uniform_24}
weighted_hostfile=${YOLO_HET_WEIGHTED_HOSTFILE:-configs/hosts_macos_cpu_weighted_24}
purpose=Compare uniform 24-rank mapping against weighted 24-rank mapping for heterogeneous Macs.
EOF

run_case "uniform_24" "${YOLO_HET_UNIFORM_HOSTFILE:-configs/hosts_macos_cpu_uniform_24}"
run_case "weighted_24" "${YOLO_HET_WEIGHTED_HOSTFILE:-configs/hosts_macos_cpu_weighted_24}"

overview="$run_dir/heterogeneous_overview.csv"
python3 - "$run_dir" "$overview" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
overview = Path(sys.argv[2])

rows = []
for label in ["uniform_24", "weighted_24"]:
    summary_path = run_dir / label / "summary.csv"
    host_metrics_path = run_dir / label / "host_metrics.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary = next(csv.DictReader(f))
    with host_metrics_path.open(newline="", encoding="utf-8") as f:
        host_rows = list(csv.DictReader(f))
    rows.append({
        "label": label,
        "world_size": summary["world_size"],
        "frames": summary["frames"],
        "tile_grid": summary["tile_grid"],
        "total_ms_with_comm": summary["total_ms_with_comm"],
        "total_ms_without_comm": summary["total_ms_without_comm"],
        "load_imbalance": summary["load_imbalance"],
        "host_task_distribution": "; ".join(
            f"{r['hostname']} ranks={r['rank_count']} tasks={r['tasks_done']} share={float(r['task_share']):.3f}"
            for r in host_rows
        ),
    })

with overview.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "label",
        "world_size",
        "frames",
        "tile_grid",
        "total_ms_with_comm",
        "total_ms_without_comm",
        "load_imbalance",
        "host_task_distribution",
    ])
    writer.writeheader()
    writer.writerows(rows)
PY

"$python_bin" scripts/report/plots/plot_heterogeneous_balance.py \
  --uniform-host-metrics "$run_dir/uniform_24/host_metrics.csv" \
  --weighted-host-metrics "$run_dir/weighted_24/host_metrics.csv" \
  --overview "$overview" \
  --output "$run_dir/figures/heterogeneous_balance.png"

echo "HETEROGENEOUS_BALANCE_DONE=YES"
echo "HETEROGENEOUS_BALANCE_DIR=$run_dir"
echo "HETEROGENEOUS_OVERVIEW=$overview"
