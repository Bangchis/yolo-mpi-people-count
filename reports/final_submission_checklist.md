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

Method 1, YOLO11n task parallelism:

- Correctness: serial vs MPI passed, 0 mismatched frames.
- Chosen N: `N=600` frames, runtime `123.667s`.
- Speedup uses `2N=1200` frames.
- Speedup at `P=12`: `1.939x` with communication, `2.011x` without communication.
- Dynamic scheduling is not always faster; it has dispatch overhead but is more general for irregular tasks and heterogeneous machines.
- YOLO-vs-ground-truth error is model accuracy, not parallel correctness.

Method 2, VGG11 no-BN distributed convolution extension:

- Code path: `src/vgg11_mpi.cpp` and `src/vgg11_mpi/`.
- Main note/runbook: `reports/method2_vgg11_notes.md`.
- Local quick suite already verifies blocking/non-blocking correctness with `max_abs_error=0`.
- Small real-image Method 2 smoke test uses CIFAR-10 `test_batch.bin` only:

```bash
VGG_CIFAR_COUNT=16 VGG_NP=2 bash scripts/run/vgg11_cifar10_local_smoke.sh
```

- Before the team starts the longer Method 2 run, run:

```bash
bash scripts/run/vgg11_method2_preflight.sh
```

- Before rewriting the final PDF around Method 2, run the three-machine report suite:

```bash
VGG_REPORT_DIR=results/vgg11_method2_report_$(date +%Y%m%d-%H%M%S) \
VGG_USE_HOSTFILE=1 \
VGG_HOSTFILE=configs/hosts_macos_core_weighted_12_4_6_2 \
MPI_MAP_BY=slot \
VGG_GRID=3x4 \
VGG_HALO_MODES="blocking nonblocking" \
VGG_SIZE_LIST="32 64 128" \
VGG_INPUT_NP=12 \
VGG_P_LIST="1 2 4 8 12" \
VGG_SPEEDUP_SIZE=64 \
VGG_REPORT_PROFILE=small \
VGG_RUN_TOPOLOGY=1 \
bash scripts/run/vgg11_report_experiments.sh
```

- Required Method 2 figures/tables: input size plot, speedup/efficiency plot, blocking vs non-blocking runtime, rank/load metrics, topology-aware vs round-robin mapping comparison.
- After the Method 2 suite finishes, use `results/vgg11_method2_report_<timestamp>/summary_tables.md` as the source for report tables.
