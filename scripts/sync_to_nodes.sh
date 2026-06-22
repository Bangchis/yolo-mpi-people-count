#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${1:-$repo_root/configs/cluster_macos.env}"

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy configs/cluster_macos.env.example first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$env_file"

remote_dir="${REMOTE_REPO_DIR:-/Users/Shared/yolo-mpi-people-count}"
targets=("${NODE1_HOST:-node1}" "${NODE2_HOST:-node2}")

repo_root_real="$(cd "$repo_root" && pwd)"
remote_dir_local="$remote_dir"
if [[ "${YOLO_SYNC_MASTER:-1}" == "1" ]]; then
  mkdir -p "$remote_dir_local"
  remote_dir_real="$(cd "$remote_dir_local" && pwd)"
  if [[ "$remote_dir_real" != "$repo_root_real" ]]; then
    echo "SYNC_HOST=master-local"
    rsync -az --delete \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude 'results/' \
      --exclude 'runs/' \
      --exclude '__pycache__/' \
      "$repo_root/" "$remote_dir_real/"
  fi
fi

for host in "${targets[@]}"; do
  echo "SYNC_HOST=$host"
  ssh "$host" "mkdir -p '$remote_dir'"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude 'results/' \
    --exclude 'runs/' \
    --exclude '__pycache__/' \
    "$repo_root/" "$host:$remote_dir/"
done

echo "SYNC_TO_NODES_DONE=YES"
