#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

report_md="${YOLO_FINAL_REPORT_MD:-reports/final_report_hust_style.md}"
report_html="${YOLO_FINAL_REPORT_HTML:-reports/final_report_hust_style.html}"
report_pdf="${YOLO_FINAL_REPORT_PDF:-}"
checklist="${YOLO_FINAL_CHECKLIST:-reports/final_submission_checklist.md}"
audit="${YOLO_REQUIREMENT_AUDIT:-reports/requirement_coverage_audit.md}"
defense_notes="${YOLO_DEFENSE_NOTES:-reports/defense_qa_notes.md}"
summary="${YOLO_RESULT_SUMMARY:-results/report_mot17_mini_final_20260623-154318/summary_tables.md}"

timestamp="$(date +%Y%m%d-%H%M%S)"
out_dir="${YOLO_SUBMISSION_DIR:-dist/final_submission_${timestamp}}"
zip_path="${out_dir}.zip"

missing=0
for path in "$report_md" "$checklist" "$audit" "$defense_notes" "$summary"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    missing=1
  fi
done
if [[ "$missing" != "0" ]]; then
  exit 1
fi

has_placeholders=0
if grep -q '<ID nhóm>\|<Họ tên - MSSV>' "$report_md"; then
  has_placeholders=1
  cat >&2 <<'EOF'
Report still contains identity placeholders.

Fill them first, for example:
  .venv/bin/python scripts/report/fill_report_identity.py \
    --group-id "<GROUP_ID>" \
    --members "Phạm Chí Bằng - 2035477; Nguyễn Thanh Lâm - 20235519; Nguyễn Phú An - 20235466; Lưu Hiếu An - 202400093"

If you intentionally want a draft package, rerun with:
  YOLO_ALLOW_PLACEHOLDERS=1 bash scripts/report/package_final_submission.sh
EOF
  if [[ "${YOLO_ALLOW_PLACEHOLDERS:-0}" != "1" ]]; then
    exit 2
  fi
fi

".venv/bin/python" scripts/report/render_report_html.py \
  --input "$report_md" \
  --output "$report_html"

if [[ -z "$report_pdf" ]]; then
  if [[ "$has_placeholders" == "1" ]]; then
    report_pdf="reports/final_report_hust_style_DRAFT.pdf"
  else
    report_pdf="reports/final_report_hust_style.pdf"
  fi
fi

".venv/bin/python" scripts/report/export_report_pdf_chrome.py \
  --input-html "$report_html" \
  --output-pdf "$report_pdf"

check_args=(
  --report "$report_md"
  --html "$report_html"
  --pdf "$report_pdf"
  --summary "$summary"
)
if [[ "$has_placeholders" == "1" ]]; then
  check_args+=(--allow-placeholders)
fi

".venv/bin/python" scripts/report/check_report_readiness.py "${check_args[@]}"

mkdir -p "$out_dir"
cp "$report_md" "$out_dir/"
cp "$report_html" "$out_dir/"
cp "$report_pdf" "$out_dir/"
cp "$checklist" "$out_dir/"
cp "$audit" "$out_dir/"
cp "$defense_notes" "$out_dir/"
cp "$summary" "$out_dir/"

cat > "$out_dir/README_SUBMISSION.txt" <<EOF
YOLO MPI People Count final submission package
Generated: $timestamp

Draft status: $([[ "$has_placeholders" == "1" ]] && echo "DRAFT - identity placeholders still exist" || echo "FINAL - no identity placeholders detected")

Open final_report_hust_style.html in a browser and Save as PDF:
  Cmd + P -> Save as PDF

PDF export included:
  $(basename "$report_pdf")

Main report source:
  final_report_hust_style.md

Evidence summary:
  summary_tables.md

Requirement coverage:
  requirement_coverage_audit.md

Defense notes:
  defense_qa_notes.md
EOF

rm -f "$zip_path"
(cd "$(dirname "$out_dir")" && zip -qr "$(basename "$zip_path")" "$(basename "$out_dir")")

echo "SUBMISSION_DIR=$out_dir"
echo "SUBMISSION_ZIP=$zip_path"
