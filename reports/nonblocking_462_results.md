# Non-Blocking MPI Experiment: Static YOLO Pipeline With 4/6/2 Placement

This note summarizes the new non-blocking communication variant for the final
Method 1 pipeline.

## Implementation Summary

The original static scheduler used blocking `MPI_Gather` and `MPI_Gatherv` to
collect variable-length detection payloads at rank 0. The new mode keeps the
same task decomposition and the same static task mapping, but replaces the final
static gather with a non-blocking two-phase gather:

1. each non-root rank sends its payload length using `MPI_Isend`;
2. rank 0 posts `MPI_Irecv` requests for all payload lengths;
3. rank 0 allocates the final receive buffer;
4. each non-root rank sends its serialized payload using `MPI_Isend`;
5. rank 0 posts `MPI_Irecv` requests for all payloads and waits for them
   together using `MPI_Waitall`.

This is a non-blocking communication variant for the static result-collection
stage. It does not yet overlap YOLO inference with communication, because each
rank must finish local detection before it has a complete serialized result to
send. It is still useful for the report because it demonstrates a clear
blocking vs non-blocking MPI communication design at the result aggregation
stage.

## Cluster Placement

The experiment uses the weighted static hostfile:

```text
master slots=4
node1 slots=6
node2 slots=2
```

This corresponds to the measured heterogeneous setup:

- master: MacBook Air M4, 4 ranks;
- node1: MacBook Pro M4, 6 ranks;
- node2: MacBook Air M2, 2 ranks.

## Command

```bash
YOLO_NB_COMM_MODES=nonblocking \
YOLO_NB_FIND_FRAME_LIST="300 600" \
YOLO_NB_QUICK=0 \
bash scripts/run/nonblocking_462_experiments.sh
```

Output folder:

```text
results/nonblocking_462_20260624-223546/
```

## Main Results

### Correctness

| Check | Value |
|---|---:|
| Correctness pass | YES |
| Frames compared | 30 |
| Mismatched frames | 0 |
| Max absolute error | 0 |
| Mean absolute error | 0.000000 |

### Accuracy Against MOT17 Ground Truth

| Metric | Value |
|---|---:|
| Frames compared | 300 |
| MAE | 8.346667 |
| RMSE | 8.676405 |
| Exact-match rate | 0.000000 |
| Mean GT count | 16.206667 |
| Mean predicted count | 7.860000 |

### Input Size Selection

| Frames | Runtime with communication | Runtime without communication | Processes | Tile grid | Communication mode |
|---:|---:|---:|---:|---|---|
| 300 | 17.614 s | 15.295 s | 12 | 5x4 | nonblocking |
| 600 | 35.452 s | 33.181 s | 12 | 5x4 | nonblocking |

With the new non-blocking static gather and 4/6/2 placement, the 600-frame run
is much shorter than the earlier dynamic report run. This is useful as an
additional optimization experiment, but the original 2-3 minute `N` result
should still be kept in the final report as the conservative course benchmark.

### Granularity And Load Balance

| Tile grid | Tasks | Max compute | Min compute | Avg compute | Total comm | Total idle | Idle gap | Balance |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2x2 | 2400 | 9.370 s | 4.828 s | 6.887 s | 17.424 s | 29.804 s | 0.485 | NO |
| 4x3 | 7200 | 28.865 s | 13.354 s | 21.002 s | 73.916 s | 94.348 s | 0.537 | NO |
| 5x4 | 12000 | 45.376 s | 20.861 s | 32.155 s | 164.678 s | 158.650 s | 0.540 | NO |

The non-blocking gather reduces aggregation cost, but it does not completely
solve load imbalance. The remaining imbalance mostly comes from heterogeneous
CPU speeds and uneven YOLO tile difficulty.

### Speedup On 1200 Frames

| Processes | Runtime with communication | Runtime without communication | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 264.179 s | 262.334 s | 1.000 | 1.000 |
| 2 | 230.918 s | 228.744 s | 1.144 | 0.572 |
| 4 | 227.601 s | 225.052 s | 1.161 | 0.290 |
| 8 | 121.852 s | 118.950 s | 2.168 | 0.271 |
| 12 | 79.059 s | 76.345 s | 3.342 | 0.278 |

The `P=12` non-blocking static run reaches `3.342x` wall-clock speedup. This is
stronger than the earlier dynamic report run because the final version combines
three improvements: static scheduling, weighted 4/6/2 placement, and
non-blocking result collection.

## Main Figures

- Runtime vs input size:
  `results/nonblocking_462_20260624-223546/nonblocking/find_N/figures/find_N_runtime.png`
- Count error plot:
  `results/nonblocking_462_20260624-223546/nonblocking/accuracy/count_error_plot.png`
- Granularity overview:
  `results/nonblocking_462_20260624-223546/nonblocking/granularity/granularity_overview.png`
- Per-rank stacked timing, 5x4:
  `results/nonblocking_462_20260624-223546/nonblocking/granularity/grid_5x4/rank_metrics_stacked.png`
- Speedup plot:
  `results/nonblocking_462_20260624-223546/nonblocking/speedup/figures/speedup.png`
- Communication-mode overview:
  `results/nonblocking_462_20260624-223546/figures/comm_mode_comparison.png`

## How To Explain This In The Report

The non-blocking variant should be presented as an additional communication
optimization of the static MPI pipeline. It should not replace the core
parallel algorithm description. The core algorithm remains:

```text
video -> frames -> tiles -> static MPI task mapping -> local YOLO inference
-> non-blocking result gather -> global duplicate removal -> people count
```

The important claim is:

```text
The final optimized configuration combines weighted process placement and
non-blocking result collection. This improves the measured P=12 speedup on the
1200-frame workload, while preserving serial-vs-MPI correctness.
```
