# Method 2: VGG11 No-BN Distributed Convolution

This note is the working source of truth for the second parallel algorithm.
Method 1 remains the YOLO11n frame/tile task-parallel pipeline. Method 2 is a
separate fine-grained data-parallel CNN algorithm.

## Report Structure Target

The final report should follow this structure, with VGG11 used instead of
VGG16 because the implementation deliberately uses VGG11 without BatchNorm:

```text
1. Introduction

2. Problem Description
   2.1 CNN-based Computer Vision Inference
   2.2 YOLO11n Video Inference
   2.3 VGG11 Convolution Layers

3. Method 1: Task Parallelism for YOLO11n Video Inference
   3.1 Data Decomposition
   3.2 1D Task Mapping
   3.3 Communication Strategy
   3.4 Load Balancing
   3.5 Parallel Algorithm

4. Method 2: Data Parallelism for VGG11 Convolution
   4.1 Data Decomposition
   4.2 2D Block Mapping
   4.3 Halo Exchange
   4.4 Blocking Communication
   4.5 Non-blocking Communication
   4.6 Load Balancing
   4.7 Parallel Algorithm

5. Experimental Setup

6. Results and Discussion
   6.1 Correctness Verification
   6.2 Input Size Selection
   6.3 Granularity and Load Balance
   6.4 Speedup and Efficiency
   6.5 Blocking vs Non-blocking Communication

7. Conclusion

8. Member Contributions
```

## Purpose

Method 2 demonstrates data parallelism inside CNN convolution layers. Instead
of assigning independent video tiles to ranks, the feature map of each VGG11
convolution layer is decomposed into 2D blocks. Each MPI rank owns one block and
must exchange halo data with neighboring ranks before computing convolution.

The model topology follows VGG11 without BatchNorm:

```text
Conv -> ReLU -> MaxPool
Conv -> ReLU -> MaxPool
Conv -> ReLU
Conv -> ReLU -> MaxPool
Conv -> ReLU
Conv -> ReLU -> MaxPool
Conv -> ReLU
Conv -> ReLU -> MaxPool
```

The benchmark supports scaled channel profiles:

```text
tiny  = small channels for local smoke tests
small = moderate channels for cluster experiments
full  = VGG11 channel sizes without BatchNorm
```

## Parallel Algorithm

For each 3x3 convolution layer:

```text
1. Arrange P MPI ranks into a Pr x Pc 2D grid.
2. Split the input feature map by H x W into 2D blocks.
3. Rank 0 scatters each core block to the matching rank.
4. Neighboring ranks exchange 1-pixel halo rows, columns, and corners.
5. Each rank computes convolution only for its local core block.
6. Rank 0 gathers local output blocks into the full feature map.
7. ReLU and MaxPool are applied on rank 0 before the next layer.
```

This is intentionally different from Method 1:

| Item | Method 1: YOLO11n Video | Method 2: VGG11 Conv |
| --- | --- | --- |
| Parallelism level | Task parallelism | Data parallelism |
| Decomposition | Frame/tile tasks | 2D feature-map blocks |
| Mapping | 1D task mapping | 2D block mapping |
| Communication | Master-worker result collection | Neighbor halo exchange |
| Topology | Star | 2D Cartesian/mesh |
| Correctness | Serial count vs MPI count | Serial Conv/VGG stack vs distributed stack |

## Topology-Aware Placement

Method 2 uses topology-aware process placement when running on the heterogeneous
three-Mac cluster. The 2D rank grid is row-major:

```text
P0  P1  P2  P3
P4  P5  P6  P7
P8  P9  P10 P11
```

For the measured 4/6/2 weighted placement:

```text
master: P0, P1, P2, P3
node1 : P4, P5, P6, P7, P8, P9
node2 : P10, P11
```

The goal is to reduce inter-machine halo edges:

```text
T_comm ~= E_intra * t_intra + E_inter * t_inter
```

where inter-machine communication over LAN is more expensive than communication
between ranks on the same machine. The program writes:

```text
topology_mapping.csv
topology_metrics.csv
topology_grid.txt
```

These files show which rank owns which grid cell and how many halo edges are
intra-machine vs inter-machine.

## Important Output Files

Each run writes:

```text
summary.csv              correctness and total runtime
layer_metrics.csv        per-layer scatter/halo/compute/gather time
rank_metrics.csv         per-rank timing and idle time
topology_mapping.csv     rank -> grid cell -> physical host
topology_metrics.csv     intra/inter-machine halo edge counts
topology_grid.txt        human-readable rank grid
manifest.txt             experiment configuration
```

The benchmark sweep writes:

```text
raw/vgg11_speedup.csv
figures/vgg11_conv_method2.png
```

The full report suite also writes:

```text
summary_tables.md
```

This file is the fastest source for copying Method 2 tables into the final
report.

## Local Quick Test

Fastest preflight:

```bash
bash scripts/run/vgg11_method2_preflight.sh
```

This checks build, local MPI execution, blocking/non-blocking correctness,
plots, and `summary_tables.md`.

```bash
VGG_RUN_DIR=results/vgg11_conv_quick \
VGG_USE_HOSTFILE=0 \
VGG_P_LIST="1 2" \
VGG_HEIGHT=32 \
VGG_WIDTH=32 \
VGG_PROFILE=tiny \
bash scripts/run/vgg11_conv_benchmark.sh
```

Quick report-suite smoke:

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

## Three-Machine Cluster Run

Use this one-shot report suite when the three Macs are on the same LAN again:

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

For only the speedup/communication benchmark:

```bash
VGG_RUN_DIR=results/vgg11_conv_cluster_$(date +%Y%m%d-%H%M%S) \
VGG_USE_HOSTFILE=1 \
VGG_HOSTFILE=configs/hosts_macos_core_weighted_12_4_6_2 \
MPI_MAP_BY=slot \
VGG_GRID=3x4 \
VGG_HALO_MODES="blocking nonblocking" \
VGG_P_LIST="1 2 4 8 12" \
VGG_HEIGHT=64 \
VGG_WIDTH=64 \
VGG_PROFILE=small \
bash scripts/run/vgg11_conv_benchmark.sh
```

For a stronger stress test:

```bash
VGG_HEIGHT=128 VGG_WIDTH=128 VGG_PROFILE=small
```

## Topology-Aware vs Round-Robin Mapping

This experiment is designed specifically for the Method 2 mapping discussion.
It compares:

```text
topology_aware: map-by slot, contiguous ranks from the same host stay together
round_robin   : map-by node, ranks are spread across machines round-robin
```

Run it on the three-machine LAN:

```bash
VGG_TOPO_RUN_DIR=results/vgg11_topology_mapping_$(date +%Y%m%d-%H%M%S) \
VGG_HOSTFILE=configs/hosts_macos_core_weighted_12_4_6_2 \
VGG_NP=12 \
VGG_GRID=3x4 \
VGG_HEIGHT=64 \
VGG_WIDTH=64 \
VGG_PROFILE=small \
bash scripts/run/vgg11_topology_mapping_comparison.sh
```

The main output is:

```text
topology_mapping_comparison.csv
```

It compares runtime and the number of inter-machine halo edges. This directly
supports the report claim that topology-aware mapping reduces LAN halo
communication.

## Method 2 Experiment Targets

Use these as the checklist for Method 2 results:

```text
Correctness:
  serial VGG11 convolution stack vs distributed VGG11 convolution stack
  metrics: max_abs_error, mean_abs_error, correct

Input size selection:
  run several H x W sizes, for example 32, 64, 128
  choose a size that is large enough to show communication/compute tradeoff

Granularity/load balance:
  compare different P and grid shapes where possible
  metrics: rank_metrics.csv, idle_ms, compute_ms, halo_ms

Speedup and efficiency:
  P = 1, 2, 4, 8, 12 on the three-Mac cluster
  metrics: distributed_ms, speedup, efficiency

Blocking vs non-blocking communication:
  VGG_HALO_MODES="blocking nonblocking"
  metrics: halo_ms, distributed_ms, speedup

Topology-aware mapping:
  compare map-by slot vs map-by node
  metrics: inter_machine_edges, inter_machine_edge_ratio, runtime_ms
```

## Report Wording

Use this short explanation:

```text
Method 1 studies coarse-grained task parallelism at the video inference level.
Method 2 studies fine-grained data parallelism inside CNN convolution layers.
In Method 2, the feature map is split into 2D blocks and mapped onto a 2D MPI
process grid. Because 3x3 convolution needs neighboring pixels, each rank
exchanges halo rows, columns, and corners with neighboring ranks before local
computation. Both blocking and non-blocking halo exchange are implemented. The
method also uses topology-aware process placement so that
neighboring blocks are preferably placed on the same physical machine, reducing
inter-machine halo communication over LAN.
```

## Current Implementation Scope

The current implementation is a C++17/OpenMPI VGG11 no-BN convolution-stack
benchmark. It uses deterministic synthetic input and weights, then compares the
distributed output against the serial convolution stack. This keeps correctness
fully measurable without depending on heavyweight pretrained VGG11 assets.
