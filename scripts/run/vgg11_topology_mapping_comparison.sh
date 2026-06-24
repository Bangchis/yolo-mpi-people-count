#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh

run_dir="${VGG_TOPO_RUN_DIR:-results/vgg11_topology_mapping_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_dir"/{topology_aware,round_robin}

hostfile="${VGG_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}"
np="${VGG_NP:-12}"
height="${VGG_HEIGHT:-64}"
width="${VGG_WIDTH:-64}"
profile="${VGG_PROFILE:-small}"
grid="${VGG_GRID:-3x4}"
halo_mode="${VGG_HALO_MODE:-blocking}"

run_case() {
  local label="$1"
  local map_by="$2"
  local out="$run_dir/$label"

  export MPI_MAP_BY="$map_by"
  make_mpi_prefix "$np" "$hostfile" 1 mpi_cmd

  echo "VGG11_TOPOLOGY_CASE label=$label map_by=$map_by out=$out"
  "${mpi_cmd[@]}" \
    build/vgg11_mpi \
    --height "$height" \
    --width "$width" \
    --profile "$profile" \
    --grid "$grid" \
    --halo-mode "$halo_mode" \
    --output-dir "$out" \
    --check-serial 1
}

cat > "$run_dir/manifest.txt" <<EOF
method=Method 2 topology-aware mapping comparison
hostfile=$hostfile
np=$np
grid=$grid
height=$height
width=$width
profile=$profile
halo_mode=$halo_mode
topology_aware_map_by=slot
round_robin_map_by=node
purpose=Compare contiguous topology-aware placement against round-robin placement for halo exchange locality.
EOF

run_case "topology_aware" "slot"
run_case "round_robin" "node"

"$python_bin" - "$run_dir" "$run_dir/topology_mapping_comparison.csv" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
output = Path(sys.argv[2])

rows = []
for label in ["topology_aware", "round_robin"]:
    summary_path = run_dir / label / "summary.csv"
    topology_path = run_dir / label / "topology_metrics.csv"
    with summary_path.open(newline="", encoding="utf-8") as f:
        summary = next(csv.DictReader(f))
    with topology_path.open(newline="", encoding="utf-8") as f:
        topology = next(csv.DictReader(f))

    rows.append({
        "label": label,
        "runtime_ms": summary["distributed_ms"],
        "grid": f"{summary['grid_rows']}x{summary['grid_cols']}",
        "total_edges": topology["total_edges"],
        "intra_machine_edges": topology["intra_machine_edges"],
        "inter_machine_edges": topology["inter_machine_edges"],
        "inter_machine_edge_ratio": topology["inter_machine_edge_ratio"],
        "correct": summary["correct"],
        "max_abs_error": summary["max_abs_error"],
    })

with output.open("w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "label",
        "runtime_ms",
        "grid",
        "total_edges",
        "intra_machine_edges",
        "inter_machine_edges",
        "inter_machine_edge_ratio",
        "correct",
        "max_abs_error",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "VGG11_TOPOLOGY_MAPPING_COMPARISON_DONE=YES"
echo "VGG11_TOPOLOGY_MAPPING_DIR=$run_dir"
echo "VGG11_TOPOLOGY_MAPPING_CSV=$run_dir/topology_mapping_comparison.csv"
