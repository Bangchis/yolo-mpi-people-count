#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run_dir="${VGG_PREFLIGHT_DIR:-results/vgg11_method2_preflight_$(date +%Y%m%d-%H%M%S)}"
python_bin="${VGG_PYTHON:-.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin=python3
fi

VGG_REPORT_DIR="$run_dir" \
VGG_USE_HOSTFILE=0 \
VGG_RUN_TOPOLOGY=0 \
VGG_SIZE_LIST="${VGG_PREFLIGHT_SIZE_LIST:-16 32}" \
VGG_P_LIST="${VGG_PREFLIGHT_P_LIST:-1 2}" \
VGG_INPUT_NP="${VGG_PREFLIGHT_INPUT_NP:-2}" \
VGG_HALO_MODES="${VGG_PREFLIGHT_HALO_MODES:-blocking nonblocking}" \
VGG_SPEEDUP_SIZE="${VGG_PREFLIGHT_SPEEDUP_SIZE:-32}" \
VGG_REPORT_PROFILE="${VGG_PREFLIGHT_PROFILE:-tiny}" \
bash scripts/run/vgg11_report_experiments.sh

"$python_bin" - "$run_dir" <<'PY'
import csv
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
speedup_path = run_dir / "speedup" / "raw" / "vgg11_speedup.csv"
input_path = run_dir / "input_size" / "input_size.csv"
summary_path = run_dir / "summary_tables.md"
figure_paths = [
    run_dir / "figures" / "vgg11_input_size.png",
    run_dir / "speedup" / "figures" / "vgg11_conv_method2.png",
]

missing = [str(path) for path in [speedup_path, input_path, summary_path, *figure_paths] if not path.exists()]
if missing:
    raise SystemExit("missing Method 2 preflight outputs: " + ", ".join(missing))

rows = []
for csv_path in [speedup_path, input_path]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))

bad = [row for row in rows if row.get("correct") != "YES"]
if bad:
    raise SystemExit(f"Method 2 correctness failed in {len(bad)} rows")

max_error = 0.0
for row in rows:
    max_error = max(max_error, float(row.get("max_abs_error", "0") or 0))

if max_error > 1e-3:
    raise SystemExit(f"Method 2 max_abs_error too large: {max_error}")

print(f"METHOD2_PREFLIGHT_CHECK=YES")
print(f"METHOD2_PREFLIGHT_MAX_ABS_ERROR={max_error:.8f}")
PY

echo "VGG11_METHOD2_PREFLIGHT_DONE=YES"
echo "VGG11_METHOD2_PREFLIGHT_DIR=$run_dir"
