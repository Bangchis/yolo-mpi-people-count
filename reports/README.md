# Reports Directory

Use these files for final submission and presentation.

## Main Files

The final report is now the English academic-style version. Its PDF exporter uses
a HUST/SoICT-inspired layout rather than a web-style layout.

| File | Purpose |
|---|---|
| `final_report_hust_style.md` | Main academic report source |
| `final_report_hust_style.html` | Self-contained report with embedded figures; open and print to PDF |
| `final_submission_checklist.md` | Step-by-step final submission checklist |
| `requirement_coverage_audit.md` | Instructor requirement coverage audit |
| `defense_qa_notes.md` | Short answers for presentation Q&A |
| `full_experiment_runbook.md` | Commands to rerun all experiments |
| `parallel_yolo_mpi_report.md` | Longer draft/reference report |

## Final Workflow

Fill group identity:

```bash
bash scripts/report/finalize_submission.sh --group-id "<GROUP_ID>"
```

The finalize command regenerates HTML/PDF, checks readiness, and packages the zip.
Manual fallback commands:

```bash
.venv/bin/python scripts/report/render_report_html.py \
  --input reports/final_report_hust_style.md \
  --output reports/final_report_hust_style.html
```

Check readiness:

```bash
.venv/bin/python scripts/report/check_report_readiness.py
```

Package:

```bash
bash scripts/report/package_final_submission.sh
```

The package command also exports PDF automatically. Manual fallback:

```bash
open reports/final_report_hust_style.html
```

Then use `Cmd + P -> Save as PDF`.

## Current Known Manual Step

The report is complete except for:

```text
<ID nhóm>
```

The member list has already been filled.
