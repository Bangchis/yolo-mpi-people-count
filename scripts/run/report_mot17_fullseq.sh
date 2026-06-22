#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/yolo_common.sh"

export YOLO_SOURCE="${YOLO_SOURCE:-data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4}"
export YOLO_GT_COUNTS="${YOLO_GT_COUNTS:-data/mot17-fullseq/MOT17-02-SDP-600_counts.csv}"
export YOLO_REPORT_DIR="${YOLO_REPORT_DIR:-results/report_mot17_fullseq_$(date +%Y%m%d-%H%M%S)}"

if [[ "${YOLO_REPORT_QUICK:-0}" != "1" ]]; then
  export YOLO_FIND_FRAME_LIST="${YOLO_FIND_FRAME_LIST:-50 100 200 300}"
  export YOLO_GRANULARITY_FRAMES="${YOLO_GRANULARITY_FRAMES:-300}"
  export YOLO_SPEEDUP_FRAMES="${YOLO_SPEEDUP_FRAMES:-600}"
  export YOLO_ACCURACY_FRAMES="${YOLO_ACCURACY_FRAMES:-600}"
fi

bash scripts/run/report_mot17_mini.sh
