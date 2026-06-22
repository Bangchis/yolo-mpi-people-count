# Project Overview

This project implements a parallel video inference pipeline for people counting.
It deliberately reuses the cluster conventions from
`parallel-macbook-cluster-setup`, but the runtime is macOS host OpenMPI instead
of Ubuntu VM OpenMPI so that YOLO can use PyTorch MPS.

The course-facing parallel implementation is `C++17 + OpenMPI` in
`src/yolo_mpi_cpp.cpp`. There is no Python MPI implementation. Python is used
only as a local per-rank YOLO inference worker over stdin/stdout, plus small
helpers for MPS evidence, rendering, and plotting.

Each MPI rank starts one `scripts/yolo_worker.py` process when
`--detector yolo` is selected. The rank sends task coordinates to that local
worker, receives person bounding boxes, remaps tile-local boxes to frame-global
coordinates in C++, and rank 0 performs global NMS and counting.

## Decomposition

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

## Imported Cluster Facts

- `master`, `node1`, and `node2` remain the canonical host aliases.
- `configs/hosts_macos_gpu` uses one slot per MacBook for the main MPS demo.
- `configs/hosts_macos_core` uses four slots per MacBook for process-count sweeps.
- OpenMPI is forced onto the LAN interface with `MPI_LAN_CIDR`.
