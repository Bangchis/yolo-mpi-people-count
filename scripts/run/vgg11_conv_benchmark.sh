#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh

run_dir="${VGG_RUN_DIR:-results/vgg11_conv_$(date +%Y%m%d-%H%M%S)}"
raw_dir="$run_dir/raw"
fig_dir="$run_dir/figures"
mkdir -p "$raw_dir" "$fig_dir"

p_list="${VGG_P_LIST:-1 2 4 8 12}"
height="${VGG_HEIGHT:-64}"
width="${VGG_WIDTH:-64}"
profile="${VGG_PROFILE:-tiny}"
grid="${VGG_GRID:-auto}"
halo_modes="${VGG_HALO_MODES:-${VGG_HALO_MODE:-blocking}}"
repeats="${VGG_REPEATS:-1}"
check_serial="${VGG_CHECK_SERIAL:-1}"
hostfile="${VGG_HOSTFILE:-${YOLO_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}}"
use_hostfile="${VGG_USE_HOSTFILE:-${YOLO_USE_HOSTFILE:-0}}"

cat > "$run_dir/manifest.txt" <<EOF
method=Method 2: VGG11 no-BN distributed convolution
binary=build/vgg11_mpi
height=$height
width=$width
profile=$profile
grid=$grid
halo_modes=$halo_modes
p_list=$p_list
repeats=$repeats
check_serial=$check_serial
use_hostfile=$use_hostfile
hostfile=$hostfile
purpose=Benchmark fine-grained data parallelism inside VGG11 convolution layers using 2D block mapping and halo exchange.
EOF

for halo_mode in $halo_modes; do
  for p in $p_list; do
    out="$raw_dir/$halo_mode/p_$p"
    mkdir -p "$out"

    make_mpi_prefix "$p" "$hostfile" "$use_hostfile" mpi_cmd

    echo "VGG11_METHOD2_RUN halo_mode=$halo_mode p=$p out=$out"
    "${mpi_cmd[@]}" \
      build/vgg11_mpi \
      --height "$height" \
      --width "$width" \
      --profile "$profile" \
      --grid "$grid" \
      --halo-mode "$halo_mode" \
      --output-dir "$out" \
      --check-serial "$check_serial" \
      --repeats "$repeats"
  done
done

"$python_bin" - "$raw_dir" "$raw_dir/vgg11_speedup.csv" <<'PY'
import csv
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
output = Path(sys.argv[2])

rows = []
summary_paths = sorted(
    raw_dir.glob("*/p_*/summary.csv"),
    key=lambda p: (p.parent.parent.name, int(p.parent.name.split("_")[1])),
)
for summary_path in summary_paths:
    with summary_path.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    topology_path = summary_path.parent / "topology_metrics.csv"
    with topology_path.open(newline="", encoding="utf-8") as f:
        topology = next(csv.DictReader(f))
    p = int(row["world_size"])
    runtime = float(row["distributed_ms"])
    rows.append({
        "halo_mode": row["halo_mode"],
        "p": p,
        "profile": row["profile"],
        "height": row["height"],
        "width": row["width"],
        "grid": f"{row['grid_rows']}x{row['grid_cols']}",
        "distributed_ms": f"{runtime:.6f}",
        "serial_ms": row["serial_ms"],
        "max_abs_error": row["max_abs_error"],
        "mean_abs_error": row["mean_abs_error"],
        "correct": row["correct"],
        "halo_total_edges": topology["total_edges"],
        "halo_intra_machine_edges": topology["intra_machine_edges"],
        "halo_inter_machine_edges": topology["inter_machine_edges"],
        "halo_inter_machine_edge_ratio": topology["inter_machine_edge_ratio"],
    })

baselines = {
    row["halo_mode"]: float(row["distributed_ms"])
    for row in rows
    if row["p"] == 1
}
for row in rows:
    runtime = float(row["distributed_ms"])
    baseline = baselines.get(row["halo_mode"])
    speedup = baseline / runtime if baseline and runtime > 0 else 0.0
    efficiency = speedup / row["p"] if row["p"] > 0 else 0.0
    row["speedup"] = f"{speedup:.6f}"
    row["efficiency"] = f"{efficiency:.6f}"

with output.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "p",
        "halo_mode",
        "profile",
        "height",
        "width",
        "grid",
        "distributed_ms",
        "serial_ms",
        "speedup",
        "efficiency",
        "max_abs_error",
        "mean_abs_error",
        "correct",
        "halo_total_edges",
        "halo_intra_machine_edges",
        "halo_inter_machine_edges",
        "halo_inter_machine_edge_ratio",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

"$python_bin" scripts/report/plots/plot_vgg11_conv.py \
  --speedup "$raw_dir/vgg11_speedup.csv" \
  --raw-dir "$raw_dir" \
  --output "$fig_dir/vgg11_conv_method2.png"

echo "VGG11_CONV_BENCHMARK_DONE=YES"
echo "VGG11_CONV_DIR=$run_dir"
echo "VGG11_CONV_SPEEDUP=$raw_dir/vgg11_speedup.csv"
