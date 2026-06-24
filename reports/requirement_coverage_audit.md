# Requirement Coverage Audit

This file maps the instructor's requirements to concrete report sections and evidence files.
It is a private checklist for final review, not necessarily part of the submitted report.

## Main Submission Artifacts

| Artifact | Purpose | Status |
|---|---|---|
| `reports/final_report_hust_style.md` | Main academic report source | Ready except group ID placeholder |
| `reports/final_report_hust_style.html` | Self-contained HTML with embedded figures for PDF export | Ready |
| `reports/final_report_hust_style_DRAFT.pdf` | Draft PDF export, currently 18 pages | Ready except group ID placeholder |
| `reports/final_submission_checklist.md` | Final manual submission steps | Ready |
| `reports/method2_vgg11_notes.md` | Method 2 VGG11 distributed convolution runbook and report outline | Implemented, needs 3-machine results before final PDF rewrite |
| `results/report_mot17_mini_final_20260623-154318/summary_tables.md` | Raw summary tables from experiments | Ready |
| `results/vgg11_method2_report_quick/` | Local Method 2 quick smoke evidence | Ready as local smoke only |
| `scripts/report/finalize_submission.sh` | One-command final packaging after group ID is known | Ready |

## Instructor Requirement Mapping

| Instructor requirement | Evidence in report | Runtime/data evidence |
|---|---|---|
| Report 10-20 pages, academic English style | `final_report_hust_style.md`, 5434 words, 7 figures, 12 rendered tables in HTML | PDF export currently 18 pages |
| At least 3 physical machines | Section 5.1, cluster roles table | MPI runs show hostnames from master/node1/node2 in result metrics |
| No cloud servers | Section 5.1 and 8 | Local MacBook host aliases and LAN config |
| C++17/OpenMPI parallel implementation | Title, summary, Sections 2.2, 5.3 | `build/yolo_mpi_cpp`, OpenMPI runs |
| Parallel level: task or data | Section 3.1 says task-level parallelism | Each task is one image region from one video frame |
| Decomposition technique | Section 3.2 says hybrid temporal-spatial decomposition | Video work is decomposed over time and image space |
| Mapping technique | Section 3.3 says one-dimensional flattened task mapping | Static, dynamic, and weighted mappings are explained |
| Communication strategy/topology | Section 3.4 says master-worker star topology | Master distributes tasks and gathers detection results |
| Blocking/non-blocking explanation | Sections 2.2, 3.4 | Blocking send/receive plus `MPI_Iprobe` polling |
| Load balancing | Section 3.5, Sections 7.4, 7.5, 7.7 | Dynamic scheduling, granularity sweep, weighted mapping |
| Parallel pseudo-code | Sections 4.2 and 4.3 | Static and dynamic pseudo-code included |
| Correctness of parallel result | Section 7.1 | `correctness_pass=YES`, 30 frames, 0 mismatched |
| YOLO-vs-ground-truth accuracy | Section 7.2 | MAE 7.983, RMSE 8.269 |
| Find N for 2-3 minute runtime | Section 7.3 | 600 frames, 123.667 seconds, twelve processes |
| Runtime with and without communication vs input size | Section 7.3 | `find_N_long_fullseq_5x4/raw/find_N.csv` |
| Granularity/load-balance experiment | Section 7.4 | `granularity_overview.csv`, `rank_metrics_stacked.png` |
| Stacked compute/communication/idle by process | Section 7.4 | `granularity/grid_4x3/rank_metrics_stacked.png` and other grid plots |
| Speedup for P=1,2,4,8,12 | Section 7.6 | `speedup_2N/raw/speedup.csv` |
| Speedup uses input 2N | Section 7.6 | 1200-frame benchmark video |
| Visualizations for results | Sections 7.2-7.7 | 7 figures embedded in HTML |
| Interesting topic/difficulty | Sections 1, 2, 3, 7 | YOLO + MPI + three-machine cluster + MOT17 |
| Demo runs | Section 7.9 | Live camera pipeline exists; benchmark is offline CPU/OpenMPI |
| Code size >= 250 lines per person | Section 5.4 and Section 8 | Selected files around 6710 lines |

## Method 2 Extension Coverage

Method 2 is the newer VGG11 no-BatchNorm distributed convolution extension.
It is intended to strengthen the parallel-computing content of the report by
adding fine-grained data parallelism inside CNN convolution layers.

| Instructor/report item | Method 2 evidence | Current status |
|---|---|---|
| Parallel level | Data parallelism inside CNN convolution layers | Implemented in `src/vgg11_mpi.cpp` and `src/vgg11_mpi/` |
| Decomposition | 2D feature-map block decomposition | Implemented in `src/vgg11_mpi/core/partition.hpp` |
| Mapping technique | `Pr x Pc` 2D process grid | Implemented, grid selected by `--grid auto` or `--grid 3x4` |
| Communication topology | 2D mesh / Cartesian-style neighbor communication | Implemented through rank-neighbor halo exchange |
| Halo exchange | Row, column, and corner halo exchange for 3x3 convolution | Implemented in `src/vgg11_mpi/conv/halo_exchange.hpp` |
| Blocking communication | `MPI_Sendrecv` halo exchange | Implemented with `--halo-mode blocking` |
| Non-blocking communication | `MPI_Irecv`, `MPI_Isend`, `MPI_Waitall` halo exchange | Implemented with `--halo-mode nonblocking` |
| Topology-aware placement | Contiguous rank placement and inter-machine halo edge counting | Implemented through hostfile order plus `topology_metrics.csv` |
| Correctness | Serial VGG11 conv stack vs distributed VGG11 conv stack | Local smoke passed with `max_abs_error=0` |
| Input size selection | Size sweep over H x W | Implemented in `scripts/run/vgg11_report_experiments.sh` |
| Speedup/efficiency | P sweep with blocking and nonblocking modes | Implemented in `scripts/run/vgg11_conv_benchmark.sh` |
| Granularity/load balance | Per-rank scatter/halo/compute/gather/idle metrics | Implemented through `rank_metrics.csv` and plots |
| Visualizations | Speedup/efficiency/breakdown and input-size figures | Implemented through `plot_vgg11_conv.py` and `plot_vgg11_input_size.py` |

Three-machine Method 2 run still required before updating the final PDF:

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

## Key Numbers To Remember

| Metric | Value |
|---|---:|
| Correctness mismatched frames | 0 |
| Accuracy MAE vs MOT17 GT | 7.983 |
| Chosen N | 600 frames |
| Runtime at N with communication | 123.667 s |
| Speedup input | 1200 frames |
| Speedup at P=12 with communication | 1.939x |
| Speedup at P=12 without communication | 2.011x |
| Weighted mapping compute-only time | 30.864 s |
| Uniform mapping compute-only time | 32.268 s |

## Remaining Manual Inputs

The final report still contains this placeholder:

```text
<ID nhóm>
```

The member list is already filled:

```text
Phạm Chí Bằng - 2035477; Nguyễn Thanh Lâm - 20235519; Nguyễn Phú An - 20235466; Lưu Hiếu An - 202400093
```

Replace the group ID before exporting PDF.
