#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/yolo_common.sh"

timestamp="$(date +%Y%m%d-%H%M%S)"
run_dir="${YOLO_RUN_DIR:-results/evidence_${timestamp}}"
evidence_dir="${YOLO_EVIDENCE_DIR:-$run_dir/evidence}"
mkdir -p "$evidence_dir"

hostfile="${YOLO_HOSTFILE:-configs/hosts_macos_gpu}"
np="${YOLO_EVIDENCE_NP:-3}"
log="$evidence_dir/cluster_evidence.txt"

{
  echo "YOLO_MACOS_CLUSTER_EVIDENCE"
  date -u +"utc_time=%Y-%m-%dT%H:%M:%SZ"
  echo "repo=$(pwd)"
  echo "run_dir=$run_dir"
  echo "hostfile=$hostfile"
  echo "np=$np"
  echo "mpi_lan_cidr=${MPI_LAN_CIDR:-192.168.31.0/24}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo
  echo "LOCAL_NODE"
  bash scripts/collect_macos_node_info.sh
  echo
  echo "HOSTFILE_CONTENT"
  cat "$hostfile"
} > "$log"

cp "$hostfile" "$evidence_dir/hosts.used"
python3 scripts/summarize_host_slots.py \
  --hostfile "$hostfile" \
  --output "$evidence_dir/host_slots.csv" \
  --summary "$evidence_dir/host_slots.env" >> "$log" 2>&1 || true

if [[ "${YOLO_USE_HOSTFILE:-1}" == "1" ]]; then
  while read -r host _; do
    [[ -z "${host:-}" || "$host" == \#* ]] && continue
    {
      echo
      echo "COMMAND: ssh $host hostname"
      ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" hostname
    } >> "$log" 2>&1 || echo "SSH_FAILED host=$host" >> "$log"
  done < "$hostfile"
else
  echo "SSH_PREFLIGHT_SKIPPED=YES use_hostfile=${YOLO_USE_HOSTFILE:-1}" >> "$log"
fi

mpi=()
make_mpi_prefix "$np" "$hostfile" "${YOLO_USE_HOSTFILE:-1}" mpi
{
  echo
  echo "COMMAND: ${mpi[*]} hostname"
  "${mpi[@]}" hostname
  echo
  echo "EVIDENCE_DONE=YES"
} >> "$log" 2>&1 || {
  code=$?
  echo "MPI_HOSTNAME_FAILED exit_code=$code" >> "$log"
  exit "$code"
}

echo "EVIDENCE_LOG=$log"
