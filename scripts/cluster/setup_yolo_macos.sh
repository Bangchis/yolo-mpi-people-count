#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

env_file="${1:-configs/cluster_macos.env}"
if [[ -f "$env_file" ]]; then
  # shellcheck disable=SC1090
  source "$env_file"
fi

choose_setup_python() {
  if [[ -n "${YOLO_SETUP_PYTHON:-}" ]]; then
    printf '%s\n' "$YOLO_SETUP_PYTHON"
    return
  fi
  for candidate in \
    /opt/homebrew/bin/python3.13 \
    /opt/homebrew/bin/python3.12 \
    /opt/homebrew/bin/python3.11 \
    /opt/homebrew/opt/python@3.13/libexec/bin/python3 \
    /opt/homebrew/opt/python@3.12/libexec/bin/python3 \
    /opt/homebrew/opt/python@3.11/libexec/bin/python3 \
    python3
  do
    if command -v "$candidate" >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  printf '%s\n' python3
}

python_bin="$(choose_setup_python)"
venv_dir="${YOLO_VENV:-.venv}"

setup_one() {
  "$python_bin" -m venv "$venv_dir"
  "$venv_dir/bin/python" -m pip install --upgrade pip
  "$venv_dir/bin/python" -m pip install -e '.[yolo,helpers]'
  bash scripts/build.sh
}

if [[ "${YOLO_SETUP_REMOTE:-0}" != "1" ]]; then
  setup_one
  echo "LOCAL_YOLO_SETUP_DONE=YES"
  exit 0
fi

remote_dir="${REMOTE_REPO_DIR:-/Users/Shared/yolo-mpi-people-count}"
repo_root_real="$(cd "$repo_root" && pwd)"
if [[ -d "$remote_dir" ]]; then
  remote_dir_real="$(cd "$remote_dir" && pwd)"
  if [[ "$remote_dir_real" != "$repo_root_real" ]]; then
    echo "REMOTE_SETUP_HOST=master-local"
    (cd "$remote_dir_real" && "$python_bin" -m venv "$venv_dir" && "$venv_dir/bin/python" -m pip install --upgrade pip && "$venv_dir/bin/python" -m pip install -e '.[yolo,helpers]' && bash scripts/build.sh)
  fi
fi

for host in "${NODE1_HOST:-node1}" "${NODE2_HOST:-node2}"; do
  echo "REMOTE_SETUP_HOST=$host"
  ssh "$host" "cd '$remote_dir' && if [[ -n \"\${YOLO_SETUP_PYTHON:-}\" ]]; then py=\"\$YOLO_SETUP_PYTHON\"; elif [[ -x /opt/homebrew/bin/python3.13 ]]; then py=/opt/homebrew/bin/python3.13; elif [[ -x /opt/homebrew/bin/python3.12 ]]; then py=/opt/homebrew/bin/python3.12; elif [[ -x /opt/homebrew/bin/python3.11 ]]; then py=/opt/homebrew/bin/python3.11; elif [[ -x /opt/homebrew/opt/python@3.13/libexec/bin/python3 ]]; then py=/opt/homebrew/opt/python@3.13/libexec/bin/python3; else py=python3; fi; rm -rf '$venv_dir'; \"\$py\" -m venv '$venv_dir' && '$venv_dir/bin/python' -m pip install --upgrade pip && '$venv_dir/bin/python' -m pip install -e '.[yolo,helpers]' && bash scripts/build.sh"
done

echo "REMOTE_YOLO_SETUP_DONE=YES"
