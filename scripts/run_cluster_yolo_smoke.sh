#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo_common.sh"

if [[ ! -f "${YOLO_CLUSTER_ENV:-configs/cluster_macos.env}" ]]; then
  echo "Missing configs/cluster_macos.env. Copy configs/cluster_macos.env.example and fill macOS IP/user values first." >&2
  exit 1
fi

if [[ ! -f "${YOLO_SOURCE:-data/classroom.mp4}" ]]; then
  echo "Missing YOLO_SOURCE=${YOLO_SOURCE:-data/classroom.mp4}. Put a real video there or set YOLO_SOURCE." >&2
  exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
export YOLO_RUN_DIR="${YOLO_RUN_DIR:-results/cluster_yolo_smoke_${timestamp}}"
export YOLO_EVIDENCE_DIR="$YOLO_RUN_DIR/evidence"
export YOLO_USE_HOSTFILE="${YOLO_USE_HOSTFILE:-1}"
export YOLO_HOSTFILE="${YOLO_HOSTFILE:-configs/hosts_macos_gpu}"
export YOLO_NP="${YOLO_NP:-3}"
export YOLO_EVIDENCE_NP="${YOLO_EVIDENCE_NP:-3}"
export YOLO_DETECTOR="${YOLO_DETECTOR:-yolo}"
export YOLO_DEMO_FRAMES="${YOLO_DEMO_FRAMES:-6}"
export YOLO_TILE_GRID="${YOLO_TILE_GRID:-1x1}"
export YOLO_RENDER_VIDEO="${YOLO_RENDER_VIDEO:-1}"
mkdir -p "$YOLO_RUN_DIR" "$YOLO_EVIDENCE_DIR"

echo "CLUSTER_YOLO_SMOKE_DIR=$YOLO_RUN_DIR"

echo
echo "PHASE 1/6: local setup and model"
bash scripts/setup_yolo_macos.sh
prepare_yolo_runtime

echo
echo "PHASE 2/6: sync repo/model/video to macOS nodes"
bash scripts/sync_to_nodes.sh

if [[ "${YOLO_SKIP_REMOTE_SETUP:-0}" != "1" ]]; then
  echo
  echo "PHASE 3/6: remote setup"
  YOLO_SETUP_REMOTE=1 bash scripts/setup_yolo_macos.sh
else
  echo
  echo "PHASE 3/6: remote setup skipped"
fi

echo
echo "PHASE 4/6: cluster evidence"
bash scripts/check_cluster_macos.sh

echo
echo "PHASE 5/6: MPS evidence on all ranks"
mpi=()
make_mpi_prefix "$YOLO_EVIDENCE_NP" "$YOLO_HOSTFILE" "$YOLO_USE_HOSTFILE" mpi
rank_cmd=()
make_rank_command rank_cmd "$python_bin" scripts/check_mps.py
echo "COMMAND: ${mpi[*]} ${rank_cmd[*]}"
"${mpi[@]}" "${rank_cmd[@]}" | tee "$YOLO_EVIDENCE_DIR/mps_evidence.txt"

echo
echo "PHASE 6/6: real YOLO C++/OpenMPI correctness smoke"
bash scripts/run_demo_correctness.sh

.venv/bin/python scripts/check_final_readiness.py \
  --run-dir "$YOLO_RUN_DIR" \
  --hostfile "$YOLO_HOSTFILE" \
  --require-host master \
  --require-host node1 \
  --require-host node2

echo "CLUSTER_YOLO_SMOKE_DONE=YES"
echo "CLUSTER_YOLO_SMOKE_DIR=$YOLO_RUN_DIR"
