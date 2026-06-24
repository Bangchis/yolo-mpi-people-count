#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

bash scripts/build.sh
prepare_yolo_runtime

run_dir="${YOLO_RUN_DIR:-results/speedup_$(date +%Y%m%d-%H%M%S)}"
raw_dir="$run_dir/raw"
mkdir -p "$raw_dir"
csv="$raw_dir/speedup.csv"
echo "world_size,total_ms_with_comm,total_ms_without_comm,speedup_with_comm,speedup_without_comm,efficiency_with_comm,efficiency_without_comm,tile_grid,schedule,comm_mode" > "$csv"

baseline_with=""
baseline_without=""
hostfile="${YOLO_SWEEP_HOSTFILE:-${YOLO_CORE_HOSTFILE:-configs/hosts_macos_core}}"

for np in ${YOLO_P_LIST:-1 2 3}; do
  mpi=()
  make_mpi_prefix "$np" "$hostfile" "${YOLO_USE_HOSTFILE:-1}" mpi
  out="$raw_dir/p_${np}"
  cpp_cmd=(build/yolo_mpi_cpp
    --source "${YOLO_SOURCE:-data/classroom.mp4}"
    --model "${YOLO_MODEL:-yolo11n.pt}"
    --device "${YOLO_DEVICE:-mps}"
    --imgsz "${YOLO_IMGSZ:-512}"
    --conf "${YOLO_CONF:-0.35}"
    --iou "${YOLO_IOU:-0.50}"
    --tile-grid "${YOLO_TILE_GRID:-1x1}"
    --overlap "${YOLO_OVERLAP:-64}"
    --tile-owner-filter "${YOLO_TILE_OWNER_FILTER:-1}"
    --dedup-ios "${YOLO_DEDUP_IOS:-0.70}"
    --dedup-center "${YOLO_DEDUP_CENTER:-0.30}"
    --dedup-axis-overlap "${YOLO_DEDUP_AXIS_OVERLAP:-0.70}"
    --dedup-gap "${YOLO_DEDUP_GAP:-0.08}"
    --dedup-near-camera "${YOLO_DEDUP_NEAR_CAMERA:-0}"
    --dedup-large-area-ratio "${YOLO_DEDUP_LARGE_AREA_RATIO:-0.12}"
    --dedup-merge "${YOLO_DEDUP_MERGE:-1}"
    --schedule "${YOLO_SCHEDULE:-static}"
    --comm-mode "${YOLO_COMM_MODE:-blocking}"
    --chunk-size "${YOLO_CHUNK_SIZE:-1}"
    --stream-batch-tasks "${YOLO_STREAM_BATCH_TASKS:-20}"
    --stream-max-pending "${YOLO_STREAM_MAX_PENDING:-2}"
    --frames "${YOLO_SPEEDUP_FRAMES:-${YOLO_PERF_FRAMES:-100}}"
    --width "${YOLO_FRAME_WIDTH:-1280}"
    --height "${YOLO_FRAME_HEIGHT:-720}"
    --detector "${YOLO_DETECTOR:-yolo}"
    --python "$python_bin"
    --worker-script "${YOLO_WORKER_SCRIPT:-scripts/runtime/yolo_worker.py}"
    --cpu-fallback "${YOLO_CPU_FALLBACK:-1}"
    --sleep-ms "${YOLO_SLEEP_MS:-0}"
    --master-compute "${YOLO_MASTER_COMPUTE:-1}"
    --verify 0
    --run-id "speedup_p_${np}"
    --output "$out")
  if [[ -n "${YOLO_DETECTOR_COMMAND:-}" ]]; then
    cpp_cmd+=(--detector-command "$YOLO_DETECTOR_COMMAND")
  fi
  rank_cmd=()
  make_rank_command rank_cmd "${cpp_cmd[@]}"
  echo "COMMAND: ${mpi[*]} ${rank_cmd[*]}"
  "${mpi[@]}" "${rank_cmd[@]}"
  pull_rank_output_if_needed "$out"
  read -r with without tile schedule comm_mode < <(python3 - "$out/summary.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    row = next(csv.DictReader(f))
print(row["total_ms_with_comm"], row["total_ms_without_comm"], row["tile_grid"], row["schedule"], row.get("comm_mode", "blocking"))
PY
)
  if [[ -z "$baseline_with" ]]; then
    baseline_with="$with"
    baseline_without="$without"
  fi
  python3 - "$csv" "$np" "$with" "$without" "$baseline_with" "$baseline_without" "$tile" "$schedule" "$comm_mode" <<'PY'
import csv, sys
path, p, tw, tc, bw, bc, tile, schedule, comm_mode = sys.argv[1:]
p = int(p); tw = float(tw); tc = float(tc); bw = float(bw); bc = float(bc)
sw = bw / tw if tw else 0.0
sc = bc / tc if tc else 0.0
with open(path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([p, tw, tc, sw, sc, sw / p if p else 0.0, sc / p if p else 0.0, tile, schedule, comm_mode])
PY
done

if [[ "${YOLO_PLOT_SPEEDUP:-1}" == "1" ]]; then
  "$python_bin" scripts/report/plots/plot_speedup.py \
    --input "$csv" \
    --output "$run_dir/figures/speedup.png" || true
fi

echo "SPEEDUP_CSV=$csv"
