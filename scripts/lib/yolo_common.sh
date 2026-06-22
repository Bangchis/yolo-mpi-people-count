#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

cluster_env="${YOLO_CLUSTER_ENV:-configs/cluster_macos.env}"
if [[ -f "$cluster_env" ]]; then
  # shellcheck disable=SC1090
  source "$cluster_env"
fi

config_file="${YOLO_EXPERIMENT_ENV:-configs/yolo_mps_experiment.env}"
if [[ -f "$config_file" ]]; then
  # shellcheck disable=SC1090
  source "$config_file"
fi

python_bin="${YOLO_PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${YOLO_PYTHON:-python3}"
fi

remote_repo_dir="${REMOTE_REPO_DIR:-$repo_root}"

assign_array() {
  local out_name="$1"
  shift
  local joined=""
  local arg quoted
  for arg in "$@"; do
    printf -v quoted "%q" "$arg"
    joined+="${joined:+ }$quoted"
  done
  eval "$out_name=($joined)"
}

prepare_yolo_runtime() {
  local detector="${YOLO_DETECTOR:-yolo}"
  local source="${YOLO_SOURCE:-data/classroom.mp4}"
  if [[ "$detector" == "yolo" && "${YOLO_AUTO_DOWNLOAD:-1}" == "1" ]]; then
    "$python_bin" scripts/assets/download_model.py \
      --model "${YOLO_MODEL_NAME:-yolo11n.pt}" \
      --output "${YOLO_MODEL:-models/yolo11n.pt}"
  fi
  if [[ -f "$source" ]]; then
    local probe_output
    if probe_output="$("$python_bin" scripts/assets/probe_video.py --source "$source" 2>/dev/null)"; then
      eval "$probe_output"
      export YOLO_FRAME_WIDTH YOLO_FRAME_HEIGHT YOLO_VIDEO_FPS YOLO_VIDEO_TOTAL_FRAMES
    fi
  fi
}

make_mpi_prefix() {
  local np="$1"
  local hostfile="$2"
  local use_hostfile="${3:-${YOLO_USE_HOSTFILE:-1}}"
  local out_name="$4"
  local out=(mpirun -np "$np")
  if [[ -n "${MPI_PREFIX:-}" ]]; then
    out+=(--prefix "$MPI_PREFIX")
  fi
  if [[ "${MPI_HETERO_NODES:-0}" == "1" ]]; then
    out+=(--hetero-nodes)
  fi
  if [[ -n "${MPI_MAP_BY:-}" ]]; then
    out+=(--map-by "$MPI_MAP_BY")
  fi
  if [[ -n "${MPI_RANK_BY:-}" ]]; then
    out+=(--rank-by "$MPI_RANK_BY")
  fi
  if [[ "$use_hostfile" == "1" ]]; then
    out+=(--hostfile "$hostfile")
    out+=(--mca btl tcp,self --mca btl_tcp_if_include "${MPI_LAN_CIDR:-192.168.31.0/24}" --mca btl_tcp_disable_family 6)
    if [[ -n "${PRTE_IF_INCLUDE:-}" ]]; then
      out+=(--prtemca prte_if_include "$PRTE_IF_INCLUDE")
    fi
    if [[ -n "${PRTE_SSH_NO_TREE_SPAWN:-}" ]]; then
      out+=(--prtemca plm_ssh_no_tree_spawn "$PRTE_SSH_NO_TREE_SPAWN")
    fi
  else
    out+=(--oversubscribe --mca btl self,sm,tcp)
  fi
  assign_array "$out_name" "${out[@]}"
}

shell_join() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v quoted "%q" "$arg"
    out+="${out:+ }$quoted"
  done
  printf '%s' "$out"
}

make_rank_command() {
  local out_name="$1"
  shift
  local out=()
  if [[ "${YOLO_USE_HOSTFILE:-1}" == "1" && "${YOLO_USE_REMOTE_WDIR:-1}" == "1" ]]; then
    local cd_cmd exec_cmd
    printf -v cd_cmd "cd %q" "$remote_repo_dir"
    exec_cmd="$(shell_join "$@")"
    out=(/bin/bash -lc "$cd_cmd && exec $exec_cmd")
  else
    out=("$@")
  fi
  assign_array "$out_name" "${out[@]}"
}

pull_rank_output_if_needed() {
  local run_dir="$1"
  if [[ "${YOLO_USE_HOSTFILE:-1}" != "1" || "${YOLO_USE_REMOTE_WDIR:-1}" != "1" ]]; then
    return
  fi
  case "$run_dir" in
    /*) return ;;
  esac
  local remote_output="$remote_repo_dir/$run_dir"
  if [[ -d "$remote_output" ]]; then
    mkdir -p "$run_dir"
    rsync -az "$remote_output/" "$run_dir/"
  fi
}
