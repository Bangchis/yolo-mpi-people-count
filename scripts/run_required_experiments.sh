#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo_common.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
export YOLO_RUN_DIR="${YOLO_RUN_DIR:-results/final_${timestamp}}"
export YOLO_EVIDENCE_DIR="${YOLO_RUN_DIR}/evidence"
mkdir -p "$YOLO_RUN_DIR" "$YOLO_EVIDENCE_DIR"

echo "YOLO_FINAL_RUN_DIR=$YOLO_RUN_DIR"

echo
echo "PHASE 0/5: build C++17/OpenMPI executable"
bash scripts/build.sh

echo
echo "PHASE 1/5: macOS cluster evidence"
bash scripts/check_cluster_macos.sh

echo
echo "PHASE 2/5: MPS evidence"
mpi=()
make_mpi_prefix "${YOLO_EVIDENCE_NP:-3}" "${YOLO_HOSTFILE:-configs/hosts_macos_gpu}" "${YOLO_USE_HOSTFILE:-1}" mpi
rank_cmd=()
make_rank_command rank_cmd "$python_bin" scripts/check_mps.py
echo "COMMAND: ${mpi[*]} ${rank_cmd[*]}"
"${mpi[@]}" "${rank_cmd[@]}" | tee "$YOLO_EVIDENCE_DIR/mps_evidence.txt"

echo
echo "PHASE 3/5: correctness demo"
YOLO_RUN_DIR="$YOLO_RUN_DIR/correctness" bash scripts/run_demo_correctness.sh

echo
echo "PHASE 4/5: find N"
YOLO_RUN_DIR="$YOLO_RUN_DIR/find_N" bash scripts/run_find_N.sh

echo
echo "PHASE 5/5: speedup sweep"
YOLO_RUN_DIR="$YOLO_RUN_DIR/speedup_gpu_safe" \
YOLO_P_LIST="${YOLO_GPU_P_LIST:-1 2 3}" \
YOLO_SWEEP_HOSTFILE="${YOLO_HOSTFILE:-configs/hosts_macos_gpu}" \
bash scripts/run_speedup_sweep.sh

YOLO_RUN_DIR="$YOLO_RUN_DIR/speedup_core_saturation" \
YOLO_P_LIST="${YOLO_CORE_P_LIST:-1 2 4 8 12}" \
YOLO_SWEEP_HOSTFILE="${YOLO_CORE_HOSTFILE:-configs/hosts_macos_core}" \
bash scripts/run_speedup_sweep.sh

echo "YOLO_REQUIRED_EXPERIMENTS_DONE=YES"
echo "YOLO_FINAL_RUN_DIR=$YOLO_RUN_DIR"
