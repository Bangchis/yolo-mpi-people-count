# Full Experiment Runbook For Final Report

Run this file when all three Macs are on the same LAN.

## 0. Goal

Collect final report evidence for:

- serial vs MPI correctness
- YOLO count vs MOT17 ground truth
- finding input size `N`
- granularity/load-balance plots
- speedup for `P = 1, 2, 4, 8, 12`

Benchmark mode is **CPU**, not MPS/GPU, because the course requirement is about MPI processes/CPU cores.

## 1. Cluster Preflight

On master:

```bash
cd "/Users/bangbang/Desktop/code python/yolo-mpi-people-count"

ssh node1 hostname
ssh node2 hostname

mpirun --prefix /opt/homebrew/opt/open-mpi \
  --hetero-nodes \
  -np 3 \
  --map-by node:OVERSUBSCRIBE \
  --hostfile configs/hosts_macos_live \
  --mca btl tcp,self \
  --mca btl_tcp_if_include 172.1.0.0/24 \
  --mca btl_tcp_disable_family 6 \
  /bin/hostname
```

Expected: output contains all three hostnames.

## 2. Sync Code And Dependencies

```bash
bash scripts/cluster/write_ssh_config.sh
bash scripts/cluster/sync_to_nodes.sh
YOLO_SETUP_REMOTE=1 bash scripts/cluster/setup_yolo_macos.sh
bash scripts/cluster/check_macos.sh
```

## 3. Quick Smoke Test

This is not for final numbers. It verifies the full pipeline.

```bash
YOLO_REPORT_DIR="results/report_mot17_mini_quick_real_$(date +%Y%m%d-%H%M%S)" \
YOLO_REPORT_QUICK=1 \
YOLO_CORRECTNESS_FRAMES=3 \
YOLO_ACCURACY_FRAMES=5 \
YOLO_GRANULARITY_FRAMES=5 \
YOLO_FIND_FRAME_LIST="3 5" \
YOLO_P_LIST="1 2" \
YOLO_SPEEDUP_FRAMES=5 \
YOLO_REPORT_MPI_NP=3 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_live \
YOLO_TILE_GRID=2x2 \
MPI_MAP_BY=node:OVERSUBSCRIBE \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=256 \
YOLO_DETECTOR=yolo \
bash scripts/run/report_mot17_mini.sh
```

Acceptance:

```text
YOLO_REPORT_DONE=YES
correctness/correctness_compare.csv exists
accuracy/accuracy.csv exists
find_N/figures/find_N_runtime.png exists
granularity/grid_*/rank_metrics_stacked.png exists
speedup/figures/speedup.png exists
```

## 4. Final MOT17-Mini Run

This is the main run for the report.

```bash
YOLO_REPORT_DIR="results/report_mot17_mini_final_$(date +%Y%m%d-%H%M%S)" \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_CONF=0.35 \
YOLO_IOU=0.50 \
YOLO_TILE_GRID=4x3 \
YOLO_SCHEDULE=dynamic \
YOLO_MASTER_COMPUTE=1 \
YOLO_REPORT_MPI_NP=12 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_core \
YOLO_FIND_FRAME_LIST="30 60 100 150 220 300" \
YOLO_GRANULARITY_GRIDS="1x1 2x2 4x3 5x4" \
YOLO_P_LIST="1 2 4 8 12" \
YOLO_SPEEDUP_FRAMES=300 \
bash scripts/run/report_mot17_mini.sh
```

After it finishes:

```bash
.venv/bin/python scripts/report/summarize_report_dir.py \
  --report-dir results/report_mot17_mini_final_<timestamp>
```

Copy the generated tables from:

```text
results/report_mot17_mini_final_<timestamp>/summary_tables.md
```

into:

```text
reports/parallel_yolo_mpi_report.md
```

## 5. If Full Run Is Too Fast For N

If even 300 frames is much less than 2 minutes, run a larger `find_N` pass:

```bash
YOLO_REPORT_DIR="results/report_mot17_mini_findN_long_$(date +%Y%m%d-%H%M%S)" \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_TILE_GRID=4x3 \
YOLO_SCHEDULE=dynamic \
YOLO_REPORT_MPI_NP=12 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_core \
YOLO_FIND_FRAME_LIST="300 450 600 750 900 1200" \
YOLO_GRANULARITY_GRIDS="4x3" \
YOLO_P_LIST="1" \
YOLO_SPEEDUP_FRAMES=300 \
bash scripts/run/report_mot17_mini.sh
```

Then choose `N` from `find_N/raw/find_N.csv`.

## 6. If Full Run Is Too Slow

Use a smaller model input and fewer grids:

```bash
YOLO_IMGSZ=256
YOLO_FIND_FRAME_LIST="30 60 100 150"
YOLO_GRANULARITY_GRIDS="1x1 2x2 4x3"
YOLO_SPEEDUP_FRAMES=150
```

In the report, state that the chosen `N` is the largest feasible value under the demo time limit.

## 7. Optional Full Sequence Stress Test

Run this only after the main report data is collected.

```bash
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_REPORT_MPI_NP=12 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_core \
bash scripts/run/mot17_fullseq_accuracy_suite.sh
```

Use this for an extra table, not as the main grade-critical experiment.

## 8. Live Demo Command

This is for presentation only, not benchmark tables.

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
MPI_MAP_BY=slot:OVERSUBSCRIBE \
YOLO_USE_HOSTFILE=1 \
YOLO_NP=30 \
YOLO_LIVE_HOSTFILE=configs/hosts_macos_cpu_max_live \
YOLO_CAMERA_INDEX=0 \
YOLO_LIVE_VIEW=1 \
YOLO_VIEW_SCALE=2.4 \
YOLO_DETECTOR=yolo \
YOLO_DEVICE=cpu \
YOLO_CONF=0.40 \
YOLO_IOU=0.65 \
YOLO_LIVE_FPS=30 \
YOLO_LIVE_FRAMES=300 \
YOLO_FRAME_WIDTH=640 \
YOLO_FRAME_HEIGHT=360 \
YOLO_LIVE_IMGSZ=256 \
YOLO_LIVE_TILE_GRID=5x4 \
YOLO_LIVE_MASTER_COMPUTE=1 \
YOLO_LIVE_ANCHOR_FULL_FRAME=1 \
YOLO_LIVE_ANCHOR_POLICY=anchor-gate \
YOLO_DEDUP_NEAR_CAMERA=0 \
YOLO_DEDUP_IOS=0.80 \
YOLO_DEDUP_MERGE=0 \
bash scripts/run/live_camera_demo.sh
```

Report wording:

```text
Live camera is a demonstration of the application pipeline. The official
benchmark results are collected from offline MOT17 video on CPU/OpenMPI.
```

## 9. Report Checklist

Before submission, verify:

- [ ] `correctness_pass=YES`
- [ ] `accuracy.csv` has MAE/RMSE
- [ ] `find_N_runtime.png` exists
- [ ] `rank_metrics_stacked.png` exists for every grid
- [ ] `speedup.png` exists
- [ ] report says task-level parallelism
- [ ] report says hybrid temporal-spatial decomposition
- [ ] report explains 1D flattened task mapping
- [ ] report explains master-worker/star topology
- [ ] report explains dynamic load balancing
- [ ] report includes pseudo-code
- [ ] report includes screenshots/figures from result directory
- [ ] report does not claim MPS/GPU benchmark as CPU-core benchmark
