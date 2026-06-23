#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

members="Phạm Chí Bằng - 2035477; Nguyễn Thanh Lâm - 20235519; Nguyễn Phú An - 20235466; Lưu Hiếu An - 202400093"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/report/finalize_submission.sh --group-id "<ID_NHOM>"

This fills the final report identity, regenerates HTML/PDF, runs readiness checks,
and creates the final submission zip.
EOF
}

group_id=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group-id)
      group_id="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$group_id" ]]; then
  echo "Missing required --group-id." >&2
  usage >&2
  exit 1
fi

if [[ "$group_id" == *"<"* || "$group_id" == *">"* ]]; then
  echo "Group ID must be the real Teams/project group ID, not a placeholder." >&2
  exit 1
fi

".venv/bin/python" scripts/report/fill_report_identity.py \
  --group-id "$group_id" \
  --members "$members"

bash scripts/report/package_final_submission.sh
