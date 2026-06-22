# Parallel YOLO MPI Algorithm And Report Runbook

This document is the report-facing explanation of the current algorithm and the
commands needed to collect the results required by the course rubric.

## 1. Problem Statement

The project solves a video object-counting problem:

```text
Input  : video frames from an offline video or the master camera
Output : per-frame person count, bounding boxes, per-rank runtime metrics
```

YOLO is used as the heavy inference kernel. The parallel programming problem is
not "parallelizing YOLO internals"; it is parallelizing the video inference
pipeline around YOLO:

```text
video/frame -> frame/tile tasks -> MPI ranks -> YOLO inference -> bbox merge -> count
```

The course-facing parallel implementation uses `C++17 + OpenMPI`. The short
entrypoint is `src/yolo_mpi_cpp.cpp`, and the readable implementation sections
are under `src/yolo_mpi/`. Python does not use MPI. Python helpers are used only
for local YOLO inference, camera capture, live display, plotting, and asset
upload.

## 2. Current Parallel Algorithm

### Parallel Level

The algorithm uses task-level parallelism with hybrid data decomposition:

```text
Temporal decomposition : video -> frames
Spatial decomposition  : each frame -> tiles
Task                   : one (frame_id, tile_id, x1, y1, x2, y2)
```

For example:

```text
frames = 100
tile_grid = 3x1
num_tasks = 100 * 3 = 300
```

Each MPI rank receives a subset of these tasks. A task is independent because
YOLO inference on a tile/frame does not need other tasks.

### Process And Device Model

OpenMPI creates CPU-level operating-system processes:

```text
MPI rank -> C++ process -> scripts/runtime/yolo_worker.py -> YOLO on cpu or mps
```

OpenMPI does not directly control the GPU. If `YOLO_DEVICE=mps`, each MPI
process calls YOLO on the Apple MPS device available on its Mac. If multiple
processes on the same Mac use `mps`, they compete for the same Apple GPU.

Use these two modes differently:

```text
Report CPU benchmark : YOLO_DEVICE=cpu, P varied by CPU cores/processes
Live/MPS demo        : YOLO_DEVICE=mps, usually one GPU rank per Mac
```

### Mapping Technique

The 2D frame/tile grid is flattened into a 1D task list.

Static scheduling uses block-cyclic mapping:

```text
block_id = floor(task_id / chunk_size)
assigned_rank = block_id mod world_size
```

Dynamic scheduling uses a master-worker queue:

```text
rank 0 sends tasks to available workers
worker finishes task -> sends bbox/metrics to rank 0 -> receives next task
rank 0 can also compute when master_compute=1
```

Dynamic scheduling is preferred for the final report because YOLO task time is
not uniform. A tile with many people or occlusion is slower than an empty tile.

### Communication Strategy

Topology:

```text
star / master-worker
```

Communication:

```text
rank 0 -> worker : task metadata or live JPEG tile
worker -> rank 0 : detection payload and metric payload
rank 0           : remaps bbox, runs global NMS/de-dup, counts people
```

The implementation currently uses blocking MPI send/receive. This is acceptable
for the report because the focus is process assignment, task granularity, load
balance, and measured overhead.

### Load Balancing

Static scheduling can be imbalanced if difficult tasks are clustered on one
rank. Dynamic scheduling improves load balance because faster ranks receive more
tasks.

This matters because the cluster is heterogeneous:

```text
MacBook Air / MacBook Pro / different CPU and GPU power
```

For CPU benchmarks, stronger machines may use more slots or simply receive more
tasks via dynamic scheduling. For GPU/MPS demos, avoid many ranks per Mac unless
you intentionally want to demonstrate GPU contention.

## 3. Main Output Files

Every important run writes a result directory under `results/`.

Required files:

```text
summary.csv       : one-row run summary and total runtime
rank_metrics.csv  : per-rank tasks_done, compute_ms, comm_ms, idle_ms
frame_counts.csv  : person count per frame
bboxes.csv        : final merged bbox output
correctness.txt   : only for correctness runs
```

Use `summary.csv` for:

```text
total_ms_with_comm
total_ms_without_comm
world_size
num_tasks
tile_grid
schedule
load_imbalance
```

Use `rank_metrics.csv` for:

```text
granularity / load balance stacked bar by rank
```

## 4. Setup Commands

On the master:

```bash
cd "/Users/bangbang/Desktop/code python/yolo-mpi-people-count"

.venv/bin/python scripts/assets/download_hf_assets.py \
  --repo-id Bangchis/yolo-mpi-people-count-assets

bash scripts/build.sh
```

When all three Macs are available:

```bash
bash scripts/cluster/write_ssh_config.sh
bash scripts/cluster/sync_to_nodes.sh
YOLO_SETUP_REMOTE=1 bash scripts/cluster/setup_yolo_macos.sh
bash scripts/cluster/check_macos.sh
```

Expected cluster aliases:

```text
master
node1
node2
```

## 5. Live Camera Commands

### Master-only CPU live test

Use this when friends' nodes are unavailable:

```bash
YOLO_USE_HOSTFILE=0 \
YOLO_NP=1 \
YOLO_CAMERA_INDEX=0 \
YOLO_LIVE_VIEW=1 \
YOLO_LIVE_FRAMES=3000 \
YOLO_LIVE_TILE_GRID=1x1 \
YOLO_LIVE_IMGSZ=320 \
YOLO_FRAME_WIDTH=640 \
YOLO_FRAME_HEIGHT=360 \
YOLO_LIVE_FPS=5 \
YOLO_DEVICE=cpu \
YOLO_DETECTOR=yolo \
YOLO_TILE_OWNER_FILTER=1 \
YOLO_DEDUP_NEAR_CAMERA=0 \
YOLO_LIVE_TEMPORAL_DEDUP=0 \
bash scripts/run/live_camera_demo.sh
```

### Master-only CPU live with multiple MPI processes

This uses more CPU processes, but can duplicate close-up people if the frame is
split too much. Prefer `2x1` over `2x2` for close-up camera demos:

```bash
YOLO_USE_HOSTFILE=0 \
YOLO_NP=4 \
YOLO_CAMERA_INDEX=0 \
YOLO_LIVE_VIEW=1 \
YOLO_LIVE_FRAMES=3000 \
YOLO_LIVE_MASTER_COMPUTE=1 \
YOLO_LIVE_ANCHOR_FULL_FRAME=1 \
YOLO_LIVE_TILE_GRID=2x1 \
YOLO_LIVE_IMGSZ=320 \
YOLO_FRAME_WIDTH=640 \
YOLO_FRAME_HEIGHT=360 \
YOLO_LIVE_FPS=5 \
YOLO_DEVICE=cpu \
YOLO_DETECTOR=yolo \
YOLO_TILE_OWNER_FILTER=1 \
YOLO_DEDUP_IOS=0.65 \
YOLO_DEDUP_NEAR_CAMERA=1 \
YOLO_DEDUP_LARGE_AREA_RATIO=0.15 \
YOLO_DEDUP_CENTER=0.45 \
YOLO_DEDUP_AXIS_OVERLAP=0.65 \
YOLO_DEDUP_GAP=0.12 \
YOLO_LIVE_TEMPORAL_DEDUP=0 \
bash scripts/run/live_camera_demo.sh
```

### Three-Mac MPS live demo

Use this for the impressive presentation demo, not as the main CPU benchmark:

```bash
bash scripts/cluster/sync_to_nodes.sh

YOLO_NP=3 \
YOLO_LIVE_HOSTFILE=configs/hosts_macos_live \
YOLO_CAMERA_INDEX=0 \
YOLO_LIVE_VIEW=1 \
YOLO_LIVE_MASTER_COMPUTE=1 \
YOLO_LIVE_ANCHOR_FULL_FRAME=1 \
YOLO_LIVE_TILE_GRID=2x1 \
YOLO_LIVE_IMGSZ=416 \
YOLO_FRAME_WIDTH=960 \
YOLO_FRAME_HEIGHT=540 \
YOLO_LIVE_FPS=5 \
YOLO_DEVICE=mps \
YOLO_DETECTOR=yolo \
YOLO_TILE_OWNER_FILTER=1 \
YOLO_DEDUP_NEAR_CAMERA=0 \
YOLO_LIVE_TEMPORAL_DEDUP=0 \
bash scripts/run/live_camera_demo.sh
```

## 6. Offline Report Experiments

Live camera is good for demo, but report measurements should use a fixed video
input so every run has the same data. Recommended wording:

```text
The system supports live camera input. For reproducible benchmarking, the group
records or uses a fixed video and replays it through the same pipeline.
```

Use either:

```text
data/report_people_720p.mp4       # recommended self-recorded report video
data/smoke_people.mp4             # small smoke-test fallback
```

### A. Correctness: serial vs MPI

Purpose:

```text
Prove the MPI result is equivalent to the serial result for the same model,
video, threshold, tile grid, and NMS.
```

Command:

```bash
YOLO_RUN_DIR=results/report_final/correctness \
YOLO_SOURCE=data/report_people_720p.mp4 \
YOLO_DEVICE=cpu \
YOLO_NP=3 \
YOLO_HOSTFILE=configs/hosts_macos_gpu \
YOLO_TILE_GRID=4x3 \
YOLO_DEMO_FRAMES=60 \
YOLO_IMGSZ=416 \
YOLO_CONF=0.25 \
YOLO_IOU=0.50 \
YOLO_SCHEDULE=dynamic \
YOLO_RENDER_VIDEO=1 \
bash scripts/run/demo_correctness.sh
```

Evidence to include:

```text
results/report_final/correctness/correctness.txt
results/report_final/correctness/frame_counts.csv
results/report_final/correctness/demo/people_count_output.mp4
```

### B. Find input size N for 2-3 minutes

Purpose:

```text
Choose N so total runtime is around 120-180 seconds at the selected maximum P.
```

Use CPU mode for course benchmark:

```bash
YOLO_RUN_DIR=results/report_final/find_N \
YOLO_SOURCE=data/report_people_720p.mp4 \
YOLO_DEVICE=cpu \
YOLO_NP=12 \
YOLO_HOSTFILE=configs/hosts_macos_core \
YOLO_TILE_GRID=4x3 \
YOLO_SCHEDULE=dynamic \
YOLO_IMGSZ=416 \
YOLO_CONF=0.25 \
YOLO_FIND_FRAME_LIST="100 200 400 600 800 1000" \
bash scripts/run/find_N.sh
```

If the cluster has more than 12 physical cores and you choose a larger `P_max`,
replace `YOLO_NP=12` with that value and update the hostfile slots.

Output:

```text
results/report_final/find_N/raw/find_N.csv
```

Plot required:

```text
x-axis: input size N, usually frames or num_tasks = frames * tile_count
y-axis: total runtime
lines : with communication and without communication
```

### C. Granularity and load balance

Purpose:

```text
Show whether task granularity is fine enough to balance load across ranks.
```

Run with the chosen `N`, `P=P_max`, and compare:

```text
tile_grid=1x1   coarse
tile_grid=2x2   medium
tile_grid=4x3   fine, 12 tasks per frame
```

Example command for one granularity:

```bash
YOLO_RUN_DIR=results/report_final/granularity_4x3 \
YOLO_SOURCE=data/report_people_720p.mp4 \
YOLO_DEVICE=cpu \
YOLO_NP=12 \
YOLO_HOSTFILE=configs/hosts_macos_core \
YOLO_TILE_GRID=4x3 \
YOLO_PERF_SCHEDULE=dynamic \
YOLO_PERF_FRAMES=$N \
YOLO_IMGSZ=416 \
YOLO_CONF=0.25 \
YOLO_RENDER_VIDEO=0 \
bash scripts/run/demo_perf.sh
```

Use `rank_metrics.csv` to draw:

```text
one bar per rank
stacked segments: compute_ms + comm_ms + idle_ms
```

Conclusion rule:

```text
If idle time differs by more than 25% between ranks, granularity is not good
enough; use smaller tasks or dynamic scheduling.
```

### D. Speedup

Purpose:

```text
Measure runtime and speedup as the number of MPI processes changes.
```

Use input size `2N`:

```bash
YOLO_RUN_DIR=results/report_final/speedup \
YOLO_SOURCE=data/report_people_720p.mp4 \
YOLO_DEVICE=cpu \
YOLO_SWEEP_HOSTFILE=configs/hosts_macos_core \
YOLO_P_LIST="1 2 4 8 12" \
YOLO_TILE_GRID=4x3 \
YOLO_SCHEDULE=dynamic \
YOLO_SPEEDUP_FRAMES=$((2*N)) \
YOLO_IMGSZ=416 \
YOLO_CONF=0.25 \
bash scripts/run/speedup_sweep.sh
```

If the physical core count is larger than 12, use:

```text
YOLO_P_LIST="1 2 4 8 16 P_MAX"
```

Output:

```text
results/report_final/speedup/raw/speedup.csv
results/report_final/speedup/figures/speedup.png
```

Report both:

```text
speedup_with_comm
speedup_without_comm
efficiency_with_comm
efficiency_without_comm
```

## 7. What To Explain To The Professor

### Why this is a parallel programming project

Say:

```text
YOLO is the heavy compute kernel, but the project parallelizes the video
inference workload around YOLO. Each frame/tile is an independent MPI task.
MPI ranks execute YOLO on their assigned tasks and rank 0 merges the results.
```

Do not say:

```text
We parallelized the inside of YOLO.
```

### Why hybrid decomposition

Say:

```text
Frame-only decomposition is too coarse when the video has few frames or some
frames are much harder than others. Tile decomposition gives finer tasks, so
dynamic scheduling can balance heterogeneous machines better.
```

### Why dynamic scheduling

Say:

```text
The cluster is heterogeneous and YOLO task time depends on image content.
Dynamic scheduling lets faster ranks receive more tasks and reduces idle time.
```

### Why CPU benchmark and MPS demo are separate

Say:

```text
The course asks for process-count sweeps based on CPU cores, so the main
performance benchmark uses YOLO_DEVICE=cpu. The MPS/GPU path is used for live
demo because multiple processes on the same Mac would contend for one Apple GPU.
```

### Main difficulties and fixes

```text
1. Heterogeneous MacBooks:
   dynamic scheduling and optional weighted host slots.

2. Tile boundary duplicate detections:
   tile-owner filter, IoS NMS, and full-frame anchor in live close-up mode.

3. GPU contention:
   one MPS rank per Mac for live demo; CPU mode for process-count benchmark.

4. Reproducibility:
   fixed video input for experiments, live camera for demo.

5. Network/SSH/MPI fragility:
   hostfiles, evidence logs, sync scripts, and scripts/cluster/check_macos.sh.
```

## 8. Report Checklist

Include these artifacts in the final report:

```text
[ ] Cluster evidence: 3 physical machines, hostnames, slots, OpenMPI version
[ ] Algorithm diagram: video -> frame/tile tasks -> MPI ranks -> YOLO -> NMS
[ ] Pseudo-code for static and dynamic scheduling
[ ] Correctness table: serial vs MPI count
[ ] Find-N plot: runtime vs input size
[ ] Granularity plot: per-rank stacked compute/comm/idle
[ ] Speedup plot: P vs speedup, with and without communication
[ ] Explanation of GPU/MPS contention
[ ] Demo screenshot or short video frame
[ ] GitHub repo and Hugging Face assets link
```

## 9. Pseudo-code

### Dynamic scheduling

```text
rank 0:
    create task list from frames and tiles
    send one task to each worker
    while unfinished tasks remain:
        if master_compute enabled:
            process one local task
        receive result from any worker
        if tasks remain:
            send next task to that worker
        else:
            send stop signal
    merge all detections by frame
    run global NMS/de-dup
    write counts and metrics

worker rank:
    load YOLO worker once
    while true:
        receive task from rank 0
        if stop signal:
            break
        run YOLO on frame/tile
        send detections and metrics to rank 0
```

### Static scheduling

```text
all ranks:
    create the same task list
    for each task:
        block_id = floor(task_id / chunk_size)
        if block_id mod world_size == rank:
            run YOLO on task
    gather all local detections to rank 0

rank 0:
    merge detections
    run global NMS/de-dup
    write counts and metrics
```
