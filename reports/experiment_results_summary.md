# Experiment Results Summary For The Final Report

This note maps the measured experiments to the exact requirements from the
parallel programming project rubric. The current final scope is Method 1:
C++17/OpenMPI task-parallel YOLO11n video people counting.

## 1. Parallel Algorithm Summary

The problem is object counting in a video. The input video is divided in two
directions:

- temporal decomposition: split the video into frames;
- spatial decomposition: split each frame into tiles.

Each MPI task is one `(frame, tile)` pair. A rank runs YOLO11n on its assigned
tiles, sends the detected bounding boxes to the root rank, and the root rank
merges duplicated boxes, applies global non-maximum suppression, and writes the
final people count per frame.

The final scheduling mode is static scheduling. The tasks are mapped to ranks by
a 1D block/round-robin task assignment. The cluster is a master-worker style MPI
program: all ranks compute tasks, and rank 0 additionally collects and merges
the results. Communication is measured separately from compute time.

For heterogeneous machines, the best measured placement gives more ranks to the
stronger MacBook Pro M4 node and fewer ranks to the weaker MacBook Air M2 node.

## 2. Required Experiment Checklist

| Professor requirement | What we measured | Main metric file | Main figure |
|---|---|---|---|
| Correctness of the parallel result | Serial YOLO vs MPI YOLO frame counts | `results/report_mot17_mini_final_20260623-154318/correctness/correctness_compare.csv` | no plot needed |
| Detector quality against ground truth | YOLO counts vs MOT17 ground truth counts | `results/report_mot17_mini_final_20260623-154318/accuracy/accuracy.csv` | `results/report_mot17_mini_final_20260623-154318/accuracy/count_error_plot.png` |
| Choose input size `N` for 2-3 minutes | Runtime for 300, 600, and 837 frames | `results/report_mot17_mini_final_20260623-154318/find_N_long_fullseq_5x4/raw/find_N.csv` | `results/report_mot17_mini_final_20260623-154318/find_N_long_fullseq_5x4/figures/find_N_runtime.png` |
| Granularity/load balance | Runtime per rank for different tile grids | `results/extra_report_live_20260623-185430/granularity_N600/granularity_overview.csv` | `results/extra_report_live_20260623-185430/granularity_N600/grid_5x4/rank_metrics_stacked.png` |
| Improve load balance | Compare uniform vs weighted static placement | `results/extra_report_live_20260623-185430/weighted_static_N600/weighted_static_comparison.csv` | `results/extra_report_live_20260623-185430/weighted_static_N600/weighted_static_comparison.png` |
| Non-blocking communication | Static non-blocking result gather with `4/6/2` placement | `results/nonblocking_462_20260624-223546/comm_mode_overview.csv` | `results/nonblocking_462_20260624-223546/figures/comm_mode_comparison.png` |
| Speedup and efficiency | Run with `P = 1, 2, 4, 8, 12` | `results/report_mot17_mini_final_20260623-154318/speedup_2N/raw/speedup.csv` | `results/report_mot17_mini_final_20260623-154318/speedup_2N/figures/speedup.png` |
| Local baseline | Compare local-only and three-machine cluster | `results/extra_report_live_20260623-185430/local_vs_cluster.csv` | `results/extra_report_live_20260623-185430/local_vs_cluster_runtime.png` |
| Cluster setup evidence | MPI works across three physical MacBooks | `results/evidence_20260624-195958/evidence/cluster_evidence.txt` | no plot needed |

## 3. Correctness Verification

Parallel correctness was checked by comparing the serial YOLO output against the
MPI output on the same input, model, thresholds, and post-processing settings.

| Frames compared | Mismatched frames | Max absolute error | Mean absolute error | Result |
|---:|---:|---:|---:|---|
| 30 | 0 | 0 | 0.000000 | PASS |

This proves that the MPI task decomposition and result merging reproduce the
serial implementation for the tested configuration.

We also compared the predicted count against MOT17 ground truth to separate
algorithm correctness from detector accuracy.

| Frames | MAE | RMSE | Exact match rate | Mean GT count | Mean predicted count |
|---:|---:|---:|---:|---:|---:|
| 300 | 7.983333 | 8.269018 | 0.000000 | 16.206667 | 8.223333 |

The pretrained YOLO11n model undercounts crowded MOT17 scenes. This is a model
accuracy limitation, not an MPI correctness failure.

## 4. Input Size Selection

The project requires choosing a data size `N` such that the program runs for
about 2-3 minutes with the full process count. The measured runtime is:

| Frames | Runtime with communication | Runtime without communication | Processes | Tile grid |
|---:|---:|---:|---:|---|
| 300 | 54.764 s | 49.156 s | 12 | 5x4 |
| 600 | 123.667 s | 114.352 s | 12 | 5x4 |
| 837 | 146.316 s | 139.136 s | 12 | 5x4 |

For the report, `N = 600` frames is the clean choice because it is 123.667 s
with communication, which is 2.06 minutes. The speedup experiment then uses the
larger `2N = 1200` frame workload.

## 5. Granularity And Load Balance

Granularity was tested by changing the tile grid. More tiles create more tasks,
but they also increase communication and merging overhead.

| Tile grid | Tasks | Max compute | Min compute | Average compute | Total communication | Total idle | Idle gap | Balanced? |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 2x2 | 2400 | 14.981 s | 4.225 s | 7.056 s | 121.984 s | 95.091 s | 0.718 | NO |
| 4x3 | 7200 | 47.291 s | 11.404 s | 19.852 s | 362.261 s | 329.268 s | 0.759 | NO |
| 5x4 | 12000 | 96.959 s | 15.758 s | 34.901 s | 777.383 s | 744.691 s | 0.837 | NO |

The raw granularity experiment shows that simply increasing the number of tiles
does not automatically solve load imbalance. It can make communication and
post-processing overhead worse.

The follow-up weighted placement experiment produced a better result:

| Placement | Runtime with communication | Runtime without communication | Compute max | Compute min | Total communication | Idle gap | Balanced? |
|---|---:|---:|---:|---:|---:|---:|---|
| `master=4,node1=4,node2=4` | 40.619 s | 36.522 s | 36.522 s | 19.519 s | 87.601 s | 0.466 | NO |
| `master=3,node1=6,node2=3` | 34.712 s | 30.339 s | 30.339 s | 19.076 s | 67.660 s | 0.371 | NO |
| `master=4,node1=6,node2=2` | 29.581 s | 27.484 s | 27.484 s | 21.032 s | 19.720 s | 0.235 | YES |

The best setting was `master=4,node1=6,node2=2`. It assigns more ranks to the
MacBook Pro M4 and fewer ranks to the MacBook Air M2. The idle gap becomes
0.235, which is below the 25% threshold required in the rubric.

## 6. Non-Blocking Communication Variant

The new non-blocking experiment keeps static scheduling and the same `4/6/2`
weighted placement, but replaces the static result gather with `MPI_Isend`,
`MPI_Irecv`, and `MPI_Waitall`.

Source: `results/nonblocking_462_20260624-223546/comm_mode_overview.csv`

| Communication mode | Correctness | Accuracy MAE | N=600 runtime | N=600 without communication | P=12 speedup runtime | P=12 speedup |
|---|---|---:|---:|---:|---:|---:|
| nonblocking | YES | 8.346667 | 35.452 s | 33.181 s | 79.059 s | 3.342 |

Interpretation: the non-blocking gather preserves serial-vs-MPI correctness and
improves the optimized static/weighted configuration. It should be presented as
an additional communication optimization. It does not mean YOLO inference itself
is overlapped with communication; the non-blocking part is the final result
collection.

## 7. Speedup And Efficiency

The speedup experiment uses a larger workload and varies the number of MPI
processes.

| Processes | Runtime with communication | Runtime without communication | Speedup with communication | Speedup without communication | Efficiency with communication |
|---:|---:|---:|---:|---:|---:|
| 1 | 345.677 s | 342.467 s | 1.000 | 1.000 | 1.000 |
| 2 | 276.155 s | 272.546 s | 1.252 | 1.257 | 0.626 |
| 4 | 244.747 s | 240.905 s | 1.412 | 1.422 | 0.353 |
| 8 | 200.770 s | 194.282 s | 1.722 | 1.763 | 0.215 |
| 12 | 178.306 s | 170.255 s | 1.939 | 2.011 | 0.162 |

The speedup is positive but sublinear. This is expected because the workload has
LAN communication, root-rank merging, nonuniform machine performance, and
variable YOLO tile difficulty.

## 8. Local Baseline Versus Three-Machine Cluster

| Case | Processes | Runtime with communication | Speedup |
|---|---:|---:|---:|
| Local-only master | 1 | 786.656 s | 1.000 |
| Local-only master | 8 | 346.109 s | 2.273 |
| Three-machine cluster | 1 | 202.715 s | 1.000 |
| Three-machine cluster | 12 | 90.900 s | 2.230 |

The local-only result is useful as a baseline. The three-machine result shows
that the program runs on the required physical MPI cluster and reduces runtime
when more machines are used.

## 9. Main Points For The Report

- Parallel level: coarse-grained task parallelism.
- Decomposition: hybrid temporal and spatial decomposition.
- Mapping: 1D mapping of `(frame, tile)` tasks to MPI ranks.
- Communication: master-worker style collection; ranks compute tasks and root
  rank merges bounding boxes and counts.
- Non-blocking optimization: final static result collection can use
  `MPI_Isend`, `MPI_Irecv`, and `MPI_Waitall`.
- Load balancing: static placement is improved by weighted host slots for
  heterogeneous MacBooks.
- Correctness: MPI result matches the serial YOLO result exactly in the
  correctness test.
- Speedup: positive but sublinear because communication and merge overhead are
  significant.
- Detector accuracy: pretrained YOLO11n is fast and simple, but not perfect in
  crowded MOT17 scenes.
