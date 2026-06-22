#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${1:-$repo_root/configs/cluster_macos.env}"

if [[ ! -f "$env_file" ]]; then
  echo "Missing $env_file. Copy configs/cluster_macos.env.example first." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$env_file"

ssh_dir="$HOME/.ssh"
ssh_config="$ssh_dir/config"
mkdir -p "$ssh_dir"
touch "$ssh_config"
chmod 700 "$ssh_dir"
chmod 600 "$ssh_config"

tmp_config="$(mktemp)"
awk '
  BEGIN { skip=0 }
  /^# yolo-mpi-people-count begin$/ { skip=1; next }
  /^# yolo-mpi-people-count end$/ { skip=0; next }
  skip == 0 { print }
' "$ssh_config" > "$tmp_config"

cat >> "$tmp_config" <<EOF
# yolo-mpi-people-count begin
Host ${MASTER_HOST:-master} ${MASTER_DNS:-} ${MASTER_IP}
  HostName ${MASTER_IP}
  User ${MASTER_USER}
  IdentityFile ~/.ssh/id_ed25519
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
  SetEnv PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

Host ${NODE1_HOST:-node1} ${NODE1_DNS:-} ${NODE1_IP}
  HostName ${NODE1_IP}
  User ${NODE1_USER}
  IdentityFile ~/.ssh/id_ed25519
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
  SetEnv PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

Host ${NODE2_HOST:-node2} ${NODE2_DNS:-} ${NODE2_IP}
  HostName ${NODE2_IP}
  User ${NODE2_USER}
  IdentityFile ~/.ssh/id_ed25519
  AddKeysToAgent yes
  StrictHostKeyChecking accept-new
  SetEnv PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
# yolo-mpi-people-count end
EOF

mv "$tmp_config" "$ssh_config"
chmod 600 "$ssh_config"

echo "MACOS_SSH_CONFIG_UPDATED=YES"
echo "SSH_CONFIG=$ssh_config"
ssh -o BatchMode=yes -o ConnectTimeout=5 "${MASTER_HOST:-master}" hostname || true
ssh -o BatchMode=yes -o ConnectTimeout=5 "${NODE1_HOST:-node1}" hostname || true
ssh -o BatchMode=yes -o ConnectTimeout=5 "${NODE2_HOST:-node2}" hostname || true
