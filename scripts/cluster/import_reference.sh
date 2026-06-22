#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
old_repo="${1:-$HOME/Desktop/code python/parallel-macbook-cluster-setup}"
cd "$repo_root"

if [[ ! -d "$old_repo" ]]; then
  echo "Missing reference repo: $old_repo" >&2
  exit 1
fi

if [[ -f "$old_repo/configs/hosts" ]]; then
  cp "$old_repo/configs/hosts" configs/hosts_macos_core
fi

cat > configs/hosts_macos_gpu <<'HOSTS'
master slots=1
node1 slots=1
node2 slots=1
HOSTS

lan_cidr="192.168.31.0/24"
if [[ -f "$old_repo/configs/cluster.env" ]]; then
  old_lan="$(grep -E '^LAN_SUBNET=' "$old_repo/configs/cluster.env" | cut -d= -f2- || true)"
  if [[ -n "$old_lan" ]]; then
    lan_cidr="$old_lan"
  fi
fi

echo "IMPORTED_CLUSTER_REFERENCE=YES"
echo "OLD_REPO=$old_repo"
echo "MACOS_GPU_HOSTFILE=configs/hosts_macos_gpu"
echo "MACOS_CORE_HOSTFILE=configs/hosts_macos_core"
echo "MPI_LAN_CIDR=$lan_cidr"
echo
echo "Note: real IPs/users/keys are not copied. Put macOS host values in configs/cluster_macos.env."
