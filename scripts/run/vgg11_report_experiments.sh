#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

run_dir="${VGG_REPORT_DIR:-results/vgg11_method2_report_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_dir"/{input_size,speedup,topology,figures}

use_hostfile="${VGG_USE_HOSTFILE:-${YOLO_USE_HOSTFILE:-0}}"
hostfile="${VGG_HOSTFILE:-${YOLO_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}}"
profile="${VGG_REPORT_PROFILE:-${VGG_PROFILE:-tiny}}"
size_list="${VGG_SIZE_LIST:-32 64 128}"
p_list="${VGG_P_LIST:-1 2 4 8 12}"
input_np="${VGG_INPUT_NP:-${VGG_NP:-2}}"
grid="${VGG_GRID:-auto}"
halo_modes="${VGG_HALO_MODES:-blocking nonblocking}"
speedup_size="${VGG_SPEEDUP_SIZE:-64}"
run_topology="${VGG_RUN_TOPOLOGY:-$use_hostfile}"

cat > "$run_dir/manifest.txt" <<EOF
method=Method 2: VGG11 no-BN distributed convolution report suite
profile=$profile
size_list=$size_list
p_list=$p_list
input_np=$input_np
speedup_size=$speedup_size
grid=$grid
halo_modes=$halo_modes
use_hostfile=$use_hostfile
hostfile=$hostfile
run_topology=$run_topology
purpose=Generate Method 2 report metrics for correctness, input size, speedup, communication strategy, load balance, and topology-aware mapping.
EOF

echo "VGG11_REPORT_STAGE=input_size"
for size in $size_list; do
  VGG_RUN_DIR="$run_dir/input_size/size_$size" \
  VGG_USE_HOSTFILE="$use_hostfile" \
  VGG_HOSTFILE="$hostfile" \
  VGG_P_LIST="$input_np" \
  VGG_HALO_MODES="$halo_modes" \
  VGG_HEIGHT="$size" \
  VGG_WIDTH="$size" \
  VGG_PROFILE="$profile" \
  VGG_GRID="$grid" \
  bash scripts/run/vgg11_conv_benchmark.sh
done

"$python_bin" - "$run_dir/input_size" "$run_dir/input_size/input_size.csv" <<'PY'
import csv
import sys
from pathlib import Path

input_dir = Path(sys.argv[1])
output = Path(sys.argv[2])

rows = []
for speedup_path in sorted(input_dir.glob("size_*/raw/vgg11_speedup.csv"), key=lambda p: int(p.parents[1].name.split("_")[1])):
    with speedup_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            layer_path = speedup_path.parent / row["halo_mode"] / f"p_{row['p']}" / "layer_metrics.csv"
            layer_rows = list(csv.DictReader(layer_path.open(newline="", encoding="utf-8")))
            row["scatter_ms"] = f"{sum(float(r['scatter_ms']) for r in layer_rows):.6f}"
            row["halo_ms"] = f"{sum(float(r['halo_ms']) for r in layer_rows):.6f}"
            row["compute_ms"] = f"{sum(float(r['compute_ms']) for r in layer_rows):.6f}"
            row["gather_ms"] = f"{sum(float(r['gather_ms']) for r in layer_rows):.6f}"
            rows.append(row)

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
        "scatter_ms",
        "halo_ms",
        "compute_ms",
        "gather_ms",
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

"$python_bin" scripts/report/plots/plot_vgg11_input_size.py \
  --input "$run_dir/input_size/input_size.csv" \
  --output "$run_dir/figures/vgg11_input_size.png"

echo "VGG11_REPORT_STAGE=speedup"
VGG_RUN_DIR="$run_dir/speedup" \
VGG_USE_HOSTFILE="$use_hostfile" \
VGG_HOSTFILE="$hostfile" \
VGG_P_LIST="$p_list" \
VGG_HALO_MODES="$halo_modes" \
VGG_HEIGHT="$speedup_size" \
VGG_WIDTH="$speedup_size" \
VGG_PROFILE="$profile" \
VGG_GRID="$grid" \
bash scripts/run/vgg11_conv_benchmark.sh

if [[ "$run_topology" == "1" ]]; then
  echo "VGG11_REPORT_STAGE=topology"
  VGG_TOPO_RUN_DIR="$run_dir/topology" \
  VGG_HOSTFILE="$hostfile" \
  VGG_NP="${VGG_TOPO_NP:-12}" \
  VGG_GRID="${VGG_TOPO_GRID:-3x4}" \
  VGG_HEIGHT="$speedup_size" \
  VGG_WIDTH="$speedup_size" \
  VGG_PROFILE="$profile" \
  VGG_HALO_MODE="${VGG_TOPO_HALO_MODE:-blocking}" \
  bash scripts/run/vgg11_topology_mapping_comparison.sh
else
  echo "VGG11_REPORT_STAGE=topology_skipped"
  cat > "$run_dir/topology/SKIPPED.txt" <<EOF
Topology-aware mapping comparison was skipped because VGG_RUN_TOPOLOGY=$run_topology.
Run it on the three-machine LAN with VGG_RUN_TOPOLOGY=1.
EOF
fi

"$python_bin" scripts/report/summarize_vgg11_method2.py \
  --run-dir "$run_dir" \
  --output "$run_dir/summary_tables.md"

echo "VGG11_METHOD2_REPORT_DONE=YES"
echo "VGG11_METHOD2_REPORT_DIR=$run_dir"
echo "VGG11_METHOD2_INPUT_SIZE=$run_dir/input_size/input_size.csv"
echo "VGG11_METHOD2_SPEEDUP=$run_dir/speedup/raw/vgg11_speedup.csv"
echo "VGG11_METHOD2_SUMMARY=$run_dir/summary_tables.md"
