# Project Overview

This project implements two parallel CNN inference methods for the course
project.

Method 1 is a parallel video inference pipeline for people counting.
It deliberately reuses the cluster conventions from
`parallel-macbook-cluster-setup`, but the runtime is macOS host OpenMPI instead
of Ubuntu VM OpenMPI so that YOLO can use PyTorch MPS.

Method 2 is a VGG11 no-BatchNorm distributed convolution benchmark. It shows
fine-grained data parallelism inside CNN convolution layers with 2D block
mapping, halo exchange, blocking/non-blocking communication, and
topology-aware process placement.

The course-facing parallel implementations are `C++17 + OpenMPI`.

- Method 1 entrypoint: `src/yolo_mpi_cpp.cpp`, with implementation split under
  `src/yolo_mpi/`.
- Method 2 entrypoint: `src/vgg11_mpi.cpp`, with implementation split under
  `src/vgg11_mpi/`.

There is no Python MPI implementation. Python is used only as a local per-rank
YOLO inference worker over stdin/stdout, plus helpers for MPS evidence,
rendering, assets, plotting, and report automation.

Each MPI rank starts one `scripts/runtime/yolo_worker.py` process when
`--detector yolo` is selected. The rank sends task coordinates to that local
worker, receives person bounding boxes, remaps tile-local boxes to frame-global
coordinates in C++, and rank 0 performs global NMS and counting.

## Decomposition

Method 1:

```text
video -> frames -> optional tiles -> MPI tasks -> YOLO MPS -> global NMS -> counts
```

Static scheduling uses block-cyclic task assignment:

```text
block_id = floor(task_id / chunk_size)
rank = block_id mod world_size
```

Dynamic scheduling uses rank 0 as a task queue and ranks `1..P-1` as workers.
For `P=1`, dynamic falls back to static.

Method 2:

```text
feature map -> 2D blocks -> halo exchange -> local Conv2D -> gather -> next layer
```

The VGG11 convolution stack uses the VGG11 no-BatchNorm topology. Each 3x3
convolution layer is computed in parallel over a `Pr x Pc` MPI rank grid.
Neighboring ranks exchange one-pixel halo rows, columns, and corners before
local convolution. The implementation supports both `blocking` and
`nonblocking` halo modes.

## Imported Cluster Facts

- `master`, `node1`, and `node2` remain the canonical host aliases.
- `configs/hosts_macos_gpu` uses one slot per MacBook for the main MPS demo.
- `configs/hosts_macos_core` uses four slots per MacBook for process-count sweeps.
- `configs/hosts_macos_core_weighted_12_4_6_2` is the preferred weighted
  12-rank placement for heterogeneous CPU benchmarks and Method 2 topology
  experiments.
- OpenMPI is forced onto the LAN interface with `MPI_LAN_CIDR`.
