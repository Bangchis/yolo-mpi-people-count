#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

dataset_dir="${VGG_CIFAR_DIR:-data/vgg11-cifar10-mini}"
count="${VGG_CIFAR_COUNT:-16}"
np="${VGG_NP:-2}"
run_dir="${VGG_RUN_DIR:-results/vgg11_cifar10_local_$(date +%Y%m%d-%H%M%S)}"

"$python_bin" scripts/assets/prepare_cifar10_mini.py \
  --count "$count" \
  --output-dir "$dataset_dir"

VGG_RUN_DIR="$run_dir" \
VGG_USE_HOSTFILE=0 \
VGG_P_LIST="$np" \
VGG_HALO_MODES="${VGG_HALO_MODES:-blocking nonblocking}" \
VGG_HEIGHT=32 \
VGG_WIDTH=32 \
VGG_PROFILE="${VGG_PROFILE:-tiny}" \
VGG_GRID="${VGG_GRID:-auto}" \
VGG_INPUT_LIST="$dataset_dir/image_list.txt" \
VGG_INPUT_LIMIT="$count" \
bash scripts/run/vgg11_conv_benchmark.sh

echo "VGG11_CIFAR10_LOCAL_DONE=YES"
echo "VGG11_CIFAR10_LOCAL_DIR=$run_dir"
echo "VGG11_CIFAR10_LOCAL_DATASET=$dataset_dir"
