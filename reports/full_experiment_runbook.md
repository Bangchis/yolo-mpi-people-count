# Full Experiment Runbook For Final Report

Run this file when all three Macs are on the same LAN.

## 0. Goal

Collect final report evidence for:

- serial vs MPI correctness
- YOLO count vs MOT17 ground truth
- finding input size `N`
- granularity/load-balance plots
- static vs dynamic scheduler comparison
- speedup for `P = 1, 2, 4, 8, 12`
- uniform vs weighted mapping on heterogeneous 24-rank runs

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
YOLO_SPEEDUP_MAP_BY=node:OVERSUBSCRIBE \
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
YOLO_SPEEDUP_MAP_BY=node:OVERSUBSCRIBE \
YOLO_RUN_SCHEDULER_COMPARE=1 \
YOLO_SCHED_COMPARE_FRAMES=150 \
YOLO_RUN_HETEROGENEOUS=1 \
YOLO_HET_FRAMES=150 \
YOLO_HET_TILE_GRID=5x4 \
YOLO_HET_NP=24 \
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

If even 300 frames is much less than 2 minutes, run a larger `find_N` pass on a full MOT17 sequence. This keeps the mini dataset for correctness/accuracy, but uses a longer input to satisfy the required 2-3 minute `N`.

```bash
long_dir="results/findN_long_fullseq_$(date +%Y%m%d-%H%M%S)"

YOLO_RUN_DIR="$long_dir" \
YOLO_SOURCE=data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4 \
YOLO_MODEL=models/yolo11n.pt \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_TILE_GRID=5x4 \
YOLO_SCHEDULE=dynamic \
YOLO_NP=12 \
YOLO_USE_HOSTFILE=1 \
YOLO_HOSTFILE=configs/hosts_macos_core \
YOLO_FIND_FRAME_LIST="300 600 837" \
bash scripts/run/find_N.sh

.venv/bin/python scripts/report/plots/plot_find_n.py \
  --input "$long_dir/raw/find_N.csv" \
  --output "$long_dir/figures/find_N_runtime.png"
```

Then choose `N` from `$long_dir/raw/find_N.csv`. In the latest measured run, `N=600` with `tile_grid=5x4` took about `123.7s`, so it is a good 2-3 minute input size.

## 6. Speedup With 2N

After choosing `N=600`, create a 1200-frame benchmark video and run the speedup sweep on it.

```bash
.venv/bin/python scripts/assets/make_speedup_benchmark_video.py \
  --output data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4 \
  --frames 1200 \
  --width 960 \
  --height 540 \
  --source data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4 \
  --source data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4

mkdir -p /Users/Shared/yolo-mpi-people-count/data/mot17-benchmark
rsync -az data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4 \
  /Users/Shared/yolo-mpi-people-count/data/mot17-benchmark/
rsync -az data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4 \
  node1:/Users/Shared/yolo-mpi-people-count/data/mot17-benchmark/
rsync -az data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4 \
  node2:/Users/Shared/yolo-mpi-people-count/data/mot17-benchmark/

speed_env="/tmp/yolo_speedup2n_cluster.env"
cp configs/cluster_macos.env "$speed_env"
perl -0pi -e 's/^MPI_MAP_BY=.*/MPI_MAP_BY=node:OVERSUBSCRIBE/m' "$speed_env"

YOLO_CLUSTER_ENV="$speed_env" \
YOLO_RUN_DIR="results/report_mot17_mini_final_<timestamp>/speedup_2N" \
YOLO_SOURCE=data/mot17-benchmark/MOT17-speedup-1200_960x540.mp4 \
YOLO_MODEL=models/yolo11n.pt \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_TILE_GRID=5x4 \
YOLO_SCHEDULE=dynamic \
YOLO_MASTER_COMPUTE=1 \
YOLO_SPEEDUP_FRAMES=1200 \
YOLO_P_LIST="1 2 4 8 12" \
YOLO_USE_HOSTFILE=1 \
YOLO_SWEEP_HOSTFILE=configs/hosts_macos_core \
bash scripts/run/speedup_sweep.sh
```

Latest measured result:

```text
P=12, with communication speedup = 1.939
P=12, without communication speedup = 2.011
```

## 7. If Full Run Is Too Slow

Use a smaller model input and fewer grids:

```bash
YOLO_IMGSZ=256
YOLO_FIND_FRAME_LIST="30 60 100 150"
YOLO_GRANULARITY_GRIDS="1x1 2x2 4x3"
YOLO_SPEEDUP_FRAMES=150
```

In the report, state that the chosen `N` is the largest feasible value under the demo time limit.

## 8. Optional Full Sequence Stress Test

Run this only after the main report data is collected.

```bash
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_REPORT_MPI_NP=12 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_core \
bash scripts/run/mot17_fullseq_accuracy_suite.sh
```

Use this for an extra table, not as the main grade-critical experiment.

## 9. Heterogeneous 24-Process Balance Experiment

This is an advanced experiment for extra report quality. It compares:

```text
uniform_24  : master/node1/node2 = 8/8/8 ranks
weighted_24 : master/node1/node2 = 8/10/6 ranks
```

The weighted case gives more ranks to the stronger node and fewer ranks to the weaker node.

```bash
YOLO_RUN_DIR="results/heterogeneous_balance_final_$(date +%Y%m%d-%H%M%S)" \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_CONF=0.35 \
YOLO_IOU=0.50 \
YOLO_SCHEDULE=dynamic \
YOLO_HET_FRAMES=150 \
YOLO_HET_TILE_GRID=5x4 \
YOLO_HET_NP=24 \
bash scripts/run/heterogeneous_balance.sh
```

Use these files in the report:

```text
heterogeneous_overview.csv
figures/heterogeneous_balance.png
uniform_24/host_metrics.csv
weighted_24/host_metrics.csv
uniform_24/rank_metrics_stacked.png
weighted_24/rank_metrics_stacked.png
```

Report wording:

```text
Because the cluster is heterogeneous, a fixed equal-slot mapping can leave a
stronger machine underused or a weaker machine overloaded. We therefore added
a weighted 24-rank experiment. The mapping 8/10/6 assigns more processes to
node1, fewer to node2, and uses dynamic scheduling so faster ranks naturally
consume more tasks.
```

## 10. Scheduler Comparison

This is included in the full report command when `YOLO_RUN_SCHEDULER_COMPARE=1`.
Run it manually only if you need to regenerate that section:

```bash
YOLO_RUN_DIR="results/scheduler_comparison_final_$(date +%Y%m%d-%H%M%S)" \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_TILE_GRID=4x3 \
YOLO_SCHED_COMPARE_FRAMES=150 \
YOLO_SCHED_COMPARE_NP=12 \
YOLO_SCHED_COMPARE_HOSTFILE=configs/hosts_macos_core \
YOLO_USE_HOSTFILE=1 \
bash scripts/run/scheduler_comparison.sh
```

Use these files in the report:

```text
scheduler_comparison.csv
figures/scheduler_comparison.png
static/rank_metrics_stacked.png
dynamic/rank_metrics_stacked.png
```

Report wording:

```text
Static scheduling is simpler and has lower coordination overhead. Dynamic
scheduling is more general for irregular videos and heterogeneous machines,
but it can be slower on small inputs because rank 0 must dispatch many tasks
and receive many result payloads.
```

## 11. Live Demo Command

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

## 12. Report Checklist

Before submission, verify:

- [ ] `correctness_pass=YES`
- [ ] `accuracy.csv` has MAE/RMSE
- [ ] `find_N_runtime.png` exists
- [ ] long find_N pass reaches 2-3 minutes
- [ ] `granularity_overview.png` exists
- [ ] `rank_metrics_stacked.png` exists for every grid
- [ ] `scheduler_comparison.png` exists
- [ ] `speedup.png` exists
- [ ] `speedup_2N/figures/speedup.png` exists for P=1,2,4,8,12
- [ ] report says task-level parallelism
- [ ] report says hybrid temporal-spatial decomposition
- [ ] report explains 1D flattened task mapping
- [ ] report explains master-worker/star topology
- [ ] report explains dynamic load balancing
- [ ] report includes pseudo-code
- [ ] report includes screenshots/figures from result directory
- [ ] optional: heterogeneous 24-process experiment is included
- [ ] report does not claim MPS/GPU benchmark as CPU-core benchmark
