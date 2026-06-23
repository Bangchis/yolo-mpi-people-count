# Final Submission Checklist

## Files To Use

The main report is written in English and exported with a HUST/SoICT-inspired academic layout:
cover page, abstract, table of contents, lists of figures/tables, main sections,
references, Times-style serif typography, A4 page, and thesis-like margins.

Main report source:

```text
reports/final_report_hust_style.md
```

Self-contained HTML with embedded figures:

```text
reports/final_report_hust_style.html
```

PDF draft/final export:

```text
reports/final_report_hust_style_DRAFT.pdf
reports/final_report_hust_style.pdf
```

Main evidence summary:

```text
results/report_mot17_mini_final_20260623-154318/summary_tables.md
```

## Before Exporting

- Replace `<ID nhóm>` with the real group ID.
- Member list is already filled: `Phạm Chí Bằng - 2035477; Nguyễn Thanh Lâm - 20235519; Nguyễn Phú An - 20235466; Lưu Hiếu An - 202400093`.
- Check that the report title, subject name, and due date are correct.
- Keep benchmark wording as CPU/OpenMPI. Do not describe MPS/live camera as the official speedup benchmark.

You can fill the placeholders and regenerate HTML with:

```bash
bash scripts/report/finalize_submission.sh --group-id "<GROUP_ID>"
```

## Export To PDF

The package script exports PDF automatically with headless Chrome and checks that the PDF is 10-20 pages:

```bash
bash scripts/report/finalize_submission.sh --group-id "<GROUP_ID>"
```

If you want to export manually, open the self-contained HTML:

```bash
open reports/final_report_hust_style.html
```

Then in the browser:

```text
Cmd + P -> Save as PDF
```

Suggested output name:

```text
<GROUP_ID>_YOLO_MPI_People_Count_Report.pdf
```

## Readiness Check

Draft check, allowing placeholders:

```bash
.venv/bin/python scripts/report/check_report_readiness.py --allow-placeholders
```

Final check, after filling group/member info:

```bash
bash scripts/report/finalize_submission.sh --group-id "<GROUP_ID>"
```

The final check must print:

```text
REPORT_READY=YES
SUBMISSION_ZIP=dist/final_submission_<timestamp>.zip
```

## Evidence To Mention During Presentation

- Correctness: serial vs MPI passed, 0 mismatched frames.
- Chosen N: `N=600` frames, runtime `123.667s`.
- Speedup uses `2N=1200` frames.
- Speedup at `P=12`: `1.939x` with communication, `2.011x` without communication.
- Dynamic scheduling is not always faster; it has dispatch overhead but is more general for irregular tasks and heterogeneous machines.
- YOLO-vs-ground-truth error is model accuracy, not parallel correctness.
