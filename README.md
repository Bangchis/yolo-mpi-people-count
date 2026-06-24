# Parallel CNN Inference on 3 MacBooks

This repo is the macOS-host mirror of the existing `parallel-macbook-cluster-setup`
cluster workflow. The old repo remains untouched and is used as the source of
truth for naming, hostfile shape, MPI flags, evidence logs, reconnect workflow,
and benchmark layout.

The course-facing parallel algorithms are implemented in `C++17 + OpenMPI`.
There are two separated methods:

- Method 1: `src/yolo_mpi_cpp.cpp` and `src/yolo_mpi/` implement task
  parallelism for YOLO11n video inference.
- Method 2: `src/vgg11_mpi.cpp` and `src/vgg11_mpi/` implement data parallelism
  for VGG11 no-BatchNorm convolution layers using 2D block mapping and halo
  exchange.

There is no Python MPI path. Python is used only for detector helpers, plots,
assets, and report automation.

By default `YOLO_MASTER_COMPUTE=1`, so rank 0 on the master also runs a local
YOLO worker on `device=mps`. In dynamic mode the master acts as both coordinator
and worker: it feeds `node1`/`node2` over MPI while keeping work for its own
Apple GPU.

The final runtime mirrors the old 3-machine topology directly on macOS hosts:

```text
master -> macOS host on MacBook 1 -> MPI rank(s) -> Apple MPS
node1  -> macOS host on MacBook 2 -> MPI rank(s) -> Apple MPS
node2  -> macOS host on MacBook 3 -> MPI rank(s) -> Apple MPS
```

## Quick Setup

Required on each MacBook:

```bash
brew install open-mpi
bash scripts/build.sh
```

YOLO helper environment on each MacBook:

```bash
bash scripts/cluster/setup_yolo_macos.sh
```

The Python environment is a local detector helper only. It does not use MPI and
does not decide task scheduling. C++ ranks keep the YOLO worker alive over
stdin/stdout so the model loads once per rank.

Create the local macOS cluster config on the master machine:

```bash
cp configs/cluster_macos.env.example configs/cluster_macos.env
nano configs/cluster_macos.env
bash scripts/cluster/write_ssh_config.sh
```

Keep `REMOTE_REPO_DIR=/Users/Shared/yolo-mpi-people-count` unless you know all
three Macs use the same absolute project path.

Verify cluster and MPS:

```bash
bash scripts/cluster/check_macos.sh
mpirun -np 3 --hostfile configs/hosts_macos_gpu \
  --mca btl tcp,self --mca btl_tcp_if_include 192.168.31.0/24 --mca btl_tcp_disable_family 6 \
  .venv/bin/python scripts/cluster/check_mps.py
```

Download the pretrained model on the master, then sync to workers:

```bash
.venv/bin/python scripts/assets/download_model.py --model yolo11n.pt --output models/yolo11n.pt
bash scripts/cluster/sync_to_nodes.sh
YOLO_SETUP_REMOTE=1 bash scripts/cluster/setup_yolo_macos.sh
```

## Hugging Face Assets

GitHub intentionally does not track runtime assets such as model weights,
videos, images, build output, local cluster env files, or result logs. Put the
shareable runtime assets on Hugging Face instead:

```bash
.venv/bin/python -m pip install '.[assets]'
.venv/bin/hf auth login
.venv/bin/python scripts/assets/upload_hf_assets.py \
  --repo-id Bangchis/yolo-mpi-people-count-assets
```

The upload script includes shareable runtime assets:

- `models/yolo11n.pt`
- `data/smoke_people.mp4`
- `data/bus.jpg`
- `data/mot17-mini/*`
- `data/mot17-fullseq/*`
- `data/vgg11-tiny-images/*`

It does not upload `configs/cluster_macos.env`, SSH keys, IP-specific evidence,
`results/`, `runs/`, or `build/`. Use `--private` if the video/image assets show
people or anything sensitive.

On a fresh machine, download the same assets with:

```bash
.venv/bin/python -m pip install '.[assets]'
.venv/bin/python scripts/assets/download_hf_assets.py \
  --repo-id Bangchis/yolo-mpi-people-count-assets
```

## Demo

Put a classroom video at `data/classroom.mp4`, then run:

```bash
bash scripts/run/cluster_yolo_smoke.sh
bash scripts/cluster/sync_to_nodes.sh
bash scripts/run/demo_correctness.sh
bash scripts/run/demo_perf.sh
```

By default the C++ executable uses `YOLO_DETECTOR=yolo`, which runs real
Ultralytics YOLO inference through `scripts/runtime/yolo_worker.py`. Use
`YOLO_DETECTOR=mock` only for fast build/scheduler tests without AI
dependencies.

Outputs are written under `results/`:

- `frame_counts.csv`
- `bboxes.csv`
- `rank_metrics.csv`
- `summary.csv`
- `demo/*.mp4`
- `evidence/cluster_evidence.txt`

Video rendering is a Python/OpenCV post-process from C++ CSV output. The
parallel algorithm remains C++17/OpenMPI.

## Method 2: VGG11 Distributed Convolution

Method 2 is a fine-grained CNN parallelization experiment. It does not run
YOLO. It runs a VGG11 no-BatchNorm convolution stack where each convolution
layer is split over a 2D MPI process grid:

```text
feature map -> 2D blocks -> halo exchange -> local Conv2D -> gather -> next layer
```

It writes correctness, speedup, communication, load-balance, and topology-aware
mapping evidence:

- `summary.csv`
- `layer_metrics.csv`
- `rank_metrics.csv`
- `topology_mapping.csv`
- `topology_metrics.csv`
- `raw/vgg11_speedup.csv`
- `figures/vgg11_conv_method2.png`

Local quick report-suite check:

```bash
VGG_REPORT_DIR=results/vgg11_method2_report_quick \
VGG_USE_HOSTFILE=0 \
VGG_RUN_TOPOLOGY=0 \
VGG_SIZE_LIST="16 32" \
VGG_P_LIST="1 2" \
VGG_INPUT_NP=2 \
VGG_HALO_MODES="blocking nonblocking" \
VGG_SPEEDUP_SIZE=32 \
VGG_REPORT_PROFILE=tiny \
bash scripts/run/vgg11_report_experiments.sh
```

Small real-image local smoke test using a tiny public-image set:

```bash
VGG_IMAGE_COUNT=4 \
VGG_NP=2 \
bash scripts/run/vgg11_tiny_images_local_smoke.sh
```

This downloads a few small public images, converts them to PPM, then runs
Method 2 with `--input-list`. The benchmark still checks distributed output
against the serial VGG11 no-BN stack and writes `image_metrics.csv`.

Three-machine Method 2 report run:

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

Three-machine tiny-image benchmark, to run only after all Macs are on the
same LAN:

```bash
VGG_CLUSTER_RUN_NOW=1 \
VGG_IMAGE_COUNT=4 \
VGG_P_LIST="1 2 4 8 12" \
MPI_MAP_BY=slot \
bash scripts/run/vgg11_tiny_images_cluster_benchmark.sh
```

The Method 2 runbook is
[reports/method2_vgg11_notes.md](reports/method2_vgg11_notes.md).

## Live Camera On Master

Live mode uses the camera only on the master rank. Rank 0 captures frames, cuts
them into JPEG tiles, sends those tiles over OpenMPI, and all enabled ranks run
YOLO on their local Apple MPS devices. The master then gathers boxes, applies
tile-ownership de-duplication plus global NMS, writes CSV output, and optionally
shows a live OpenCV window.

For live camera runs, `configs/hosts_macos_live` pins rank 0 to `localhost` so
macOS camera and GUI permissions stay attached to the Terminal session on the
master machine:

```text
localhost slots=1
node1 slots=1
node2 slots=1
```

First make sure SSH aliases and remote sync work:

```bash
ssh master hostname
ssh node1 hostname
ssh node2 hostname
bash scripts/cluster/sync_to_nodes.sh
YOLO_SETUP_REMOTE=1 bash scripts/cluster/setup_yolo_macos.sh
```

Run the real camera on the master:

```bash
YOLO_NP=3 \
YOLO_LIVE_HOSTFILE=configs/hosts_macos_live \
YOLO_CAMERA_INDEX=0 \
YOLO_LIVE_FRAMES=100 \
YOLO_LIVE_VIEW=1 \
YOLO_LIVE_MASTER_COMPUTE=1 \
YOLO_LIVE_IMGSZ=416 \
YOLO_LIVE_TILE_GRID=3x1 \
YOLO_TILE_OWNER_FILTER=1 \
YOLO_DEDUP_IOS=0.70 \
YOLO_DEDUP_CENTER=0.30 \
YOLO_DEDUP_AXIS_OVERLAP=0.70 \
YOLO_DEDUP_GAP=0.08 \
YOLO_DEDUP_NEAR_CAMERA=0 \
YOLO_DEDUP_LARGE_AREA_RATIO=0.12 \
YOLO_DEDUP_MERGE=1 \
YOLO_LIVE_TEMPORAL_DEDUP=0 \
bash scripts/run/live_camera_demo.sh
```

If macOS asks for Camera permission, grant it to Terminal or the IDE terminal,
then run the command again.

If the live window lags, keep `YOLO_LIVE_IMGSZ=416`, lower the frame size, or
set `YOLO_LIVE_FPS=5`. If one person is counted multiple times near tile
borders, keep `YOLO_TILE_OWNER_FILTER=1`; for crowded scenes, tune
`YOLO_DEDUP_IOS` between `0.65` and `0.80`, or increase
`YOLO_DEDUP_CENTER` toward `0.45` for people very close to the camera.
Use `YOLO_LIVE_TEMPORAL_DEDUP=1` only when the live count flickers because it
can merge aggressively in crowded frames.

For close-up camera demos where one near person is split into multiple tile
boxes, prefer full-frame anchoring over aggressive near-camera merging:

```bash
YOLO_LIVE_ANCHOR_FULL_FRAME=1 \
YOLO_LIVE_TILE_GRID=2x1 \
YOLO_DEDUP_NEAR_CAMERA=0 \
bash scripts/run/live_camera_demo.sh
```

To test the same live pipeline without opening a camera, feed a video file as
the master source:

```bash
YOLO_USE_HOSTFILE=0 \
YOLO_NP=2 \
YOLO_LIVE_VIDEO_SOURCE=data/smoke_people.mp4 \
YOLO_LIVE_VIEW=0 \
YOLO_LIVE_FRAMES=1 \
YOLO_TILE_GRID=1x1 \
bash scripts/run/live_camera_demo.sh
```

Live outputs are written under `results/live_camera_*`:

- `live_events.csv`
- `frame_counts.csv`
- `bboxes.csv`
- `rank_metrics.csv`
- `summary.csv`
- `live_frames/*.jpg` when the viewer is enabled

## Main CLI

```bash
mpirun -np 3 --hostfile configs/hosts_macos_gpu \
  --mca btl tcp,self --mca btl_tcp_if_include 192.168.31.0/24 --mca btl_tcp_disable_family 6 \
  build/yolo_mpi_cpp \
  --source data/classroom.mp4 \
  --model models/yolo11n.pt \
  --device mps \
  --imgsz 512 \
  --tile-grid 2x2 \
  --overlap 64 \
  --conf 0.35 \
  --iou 0.50 \
  --schedule static \
  --master-compute 1 \
  --frames 20 \
  --detector yolo \
  --python .venv/bin/python \
  --worker-script scripts/runtime/yolo_worker.py \
  --verify 1 \
  --output results/demo_correctness
```

Fast scheduler-only smoke:

```bash
mpirun -np 2 --oversubscribe --mca btl self,sm,tcp \
  build/yolo_mpi_cpp --frames 4 --tile-grid 2x2 --detector mock --verify 1
```

Check a run directory before using it in the report:

```bash
.venv/bin/python scripts/report/check_final_readiness.py \
  --run-dir results/demo_correctness_YYYYMMDD-HHMMSS \
  --hostfile configs/hosts_macos_gpu \
  --require-host master --require-host node1 --require-host node2
```

## Cluster Convention Imported From The Old Repo

- Host aliases: `master`, `node1`, `node2`
- Core hostfile: `master/node1/node2 slots=4`
- GPU-safe hostfile: `master/node1/node2 slots=1`
- LAN default: `192.168.31.0/24`
- OpenMPI TCP flags:

```text
--mca btl tcp,self --mca btl_tcp_if_include "$MPI_LAN_CIDR" --mca btl_tcp_disable_family 6
```

Use `configs/hosts_macos_gpu` for the main demo. Use
`configs/hosts_macos_core` for the course-required `P=1,2,4,8,12` sweep; those
runs intentionally show GPU contention when multiple ranks share one Apple GPU.
