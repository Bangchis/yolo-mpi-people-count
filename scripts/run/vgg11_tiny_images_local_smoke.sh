#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

dataset_dir="${VGG_IMAGE_DIR:-data/vgg11-tiny-images}"
count="${VGG_IMAGE_COUNT:-4}"
size="${VGG_IMAGE_SIZE:-64}"
np="${VGG_NP:-2}"
run_dir="${VGG_RUN_DIR:-results/vgg11_tiny_images_local_$(date +%Y%m%d-%H%M%S)}"

"$python_bin" scripts/assets/prepare_vgg11_tiny_images.py \
  --limit "$count" \
  --size "$size" \
  --output-dir "$dataset_dir"

VGG_RUN_DIR="$run_dir" \
VGG_USE_HOSTFILE=0 \
VGG_P_LIST="$np" \
VGG_HALO_MODES="${VGG_HALO_MODES:-blocking nonblocking}" \
VGG_HEIGHT="$size" \
VGG_WIDTH="$size" \
VGG_PROFILE="${VGG_PROFILE:-tiny}" \
VGG_GRID="${VGG_GRID:-auto}" \
VGG_INPUT_LIST="$dataset_dir/image_list.txt" \
VGG_INPUT_LIMIT="$count" \
bash scripts/run/vgg11_conv_benchmark.sh

echo "VGG11_TINY_IMAGES_LOCAL_DONE=YES"
echo "VGG11_TINY_IMAGES_LOCAL_DIR=$run_dir"
echo "VGG11_TINY_IMAGES_LOCAL_DATASET=$dataset_dir"
