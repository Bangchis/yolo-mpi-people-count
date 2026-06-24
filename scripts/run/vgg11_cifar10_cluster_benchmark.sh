#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

dataset_dir="${VGG_CIFAR_DIR:-data/vgg11-cifar10-mini}"
count="${VGG_CIFAR_COUNT:-32}"
run_dir="${VGG_RUN_DIR:-results/vgg11_cifar10_cluster_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$run_dir"

"$python_bin" scripts/assets/prepare_cifar10_mini.py \
  --count "$count" \
  --output-dir "$dataset_dir"

cat > "$run_dir/cluster_command.sh" <<EOF
# Run this after all 3 Macs are on the same LAN and SSH/MPI preflight passes.
VGG_RUN_DIR="$run_dir" \\
VGG_USE_HOSTFILE=1 \\
VGG_HOSTFILE="\${VGG_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}" \\
VGG_P_LIST="\${VGG_P_LIST:-1 2 4 8 12}" \\
VGG_HALO_MODES="\${VGG_HALO_MODES:-blocking nonblocking}" \\
VGG_HEIGHT=32 \\
VGG_WIDTH=32 \\
VGG_PROFILE="\${VGG_PROFILE:-tiny}" \\
VGG_GRID="\${VGG_GRID:-auto}" \\
VGG_INPUT_LIST="$dataset_dir/image_list.txt" \\
VGG_INPUT_LIMIT="$count" \\
bash scripts/run/vgg11_conv_benchmark.sh
EOF

if [[ "${VGG_CLUSTER_RUN_NOW:-0}" != "1" ]]; then
  echo "VGG11_CIFAR10_CLUSTER_READY=YES"
  echo "VGG11_CIFAR10_CLUSTER_DATASET=$dataset_dir"
  echo "VGG11_CIFAR10_CLUSTER_COMMAND_FILE=$run_dir/cluster_command.sh"
  echo "Set VGG_CLUSTER_RUN_NOW=1 when all 3 Macs are on the same LAN."
  exit 0
fi

VGG_RUN_DIR="$run_dir" \
VGG_USE_HOSTFILE=1 \
VGG_HOSTFILE="${VGG_HOSTFILE:-configs/hosts_macos_core_weighted_12_4_6_2}" \
VGG_P_LIST="${VGG_P_LIST:-1 2 4 8 12}" \
VGG_HALO_MODES="${VGG_HALO_MODES:-blocking nonblocking}" \
VGG_HEIGHT=32 \
VGG_WIDTH=32 \
VGG_PROFILE="${VGG_PROFILE:-tiny}" \
VGG_GRID="${VGG_GRID:-auto}" \
VGG_INPUT_LIST="$dataset_dir/image_list.txt" \
VGG_INPUT_LIMIT="$count" \
bash scripts/run/vgg11_conv_benchmark.sh

echo "VGG11_CIFAR10_CLUSTER_DONE=YES"
echo "VGG11_CIFAR10_CLUSTER_DIR=$run_dir"
