# Additional Experiments on 23 June 2026

This note records extra experiments requested after the main report draft had
already been prepared. The goal was to strengthen the discussion around local
execution versus cluster execution, scheduling strategy, granularity, and load
balance.

The main new full-cluster output directory is:

```text
results/extra_report_live_20260623-185430/
```

The earlier partial-cluster output directory is:

```text
results/extra_report_20260623-174934/
```

## Cluster Availability

At first, node2 was unreachable from the master machine. SSH to node2 timed out
and ping showed 100 percent packet loss. During that period, only local-only and
two-machine partial-cluster experiments could be run.

Later, node2 rejoined the LAN and the full three-machine cluster became
available again. The final preflight succeeded:

- master SSH: reachable
- node1 SSH: reachable
- node2 SSH: reachable
- MPI hostname test: all three machines appeared
- C++17/OpenMPI/YOLO smoke test: completed on three ranks

The full-cluster results below should be used as the primary additional
evidence. The earlier partial-cluster results are kept only as supporting
material.

## Local-Only Baseline on Master

This experiment was run on the master machine only, using CPU execution,
600 frames, a five-by-four region grid, and dynamic scheduling.

| Processes | Runtime with communication (s) | Runtime without communication (s) | Speedup |
|---:|---:|---:|---:|
| 1 | 786.656 | 783.676 | 1.000 |
| 2 | 288.494 | 284.278 | 2.727 |
| 4 | 291.477 | 285.320 | 2.699 |
| 8 | 346.109 | 337.773 | 2.273 |

Interpretation:

- The best local-only result in this rerun was two processes.
- Increasing to four and eight local processes did not improve runtime.
- The likely causes are CPU contention, worker overhead, scheduler overhead,
  and thermal or system-load variation on a single MacBook.
- This supports the argument that process count alone is not enough; process
  placement and hardware distribution matter.

## Full Three-Machine Speedup on 600 Frames

This experiment used the three physical Mac machines, CPU execution, 600 frames,
a five-by-four region grid, and dynamic scheduling.

| Processes | Runtime with communication (s) | Runtime without communication (s) | Speedup with communication | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 202.715 | 200.470 | 1.000 | 1.000 |
| 2 | 204.701 | 201.416 | 0.990 | 0.495 |
| 4 | 244.848 | 239.662 | 0.828 | 0.207 |
| 8 | 133.360 | 127.315 | 1.520 | 0.190 |
| 12 | 90.900 | 84.188 | 2.230 | 0.186 |

Interpretation:

- The result is not linear, but the twelve-process cluster run is clearly faster
  than the single-process baseline.
- Two and four processes did not improve runtime because OpenMPI filled the
  master slots first, so those runs mainly measured local contention.
- Eight processes began using the second machine and the runtime improved.
- Twelve processes used all three machines and produced the best runtime in
  this sweep.
- This is a useful teaching point: physical distribution matters more than
  merely increasing the number of processes on one machine.

## Static versus Dynamic Scheduling on 600 Frames

This full-cluster experiment used twelve processes, 600 frames, a five-by-four
region grid, and CPU execution.

| Scheduler | Runtime with communication (s) | Runtime without communication (s) | Load imbalance | Idle-gap indicator | Balance result |
|---|---:|---:|---:|---:|---|
| Static | 40.619 | 36.522 | 1.202 | 0.466 | Not balanced |
| Dynamic | 102.316 | 96.959 | 2.778 | 0.837 | Not balanced |

Interpretation:

- Static scheduling was much faster on this workload.
- Dynamic scheduling introduced significant dispatch and communication overhead.
- Dynamic scheduling is still conceptually useful for irregular workloads, but
  this experiment shows that it is not automatically faster.
- For this YOLO task, static assignment is strong because the generated tile
  tasks are numerous and reasonably regular at this input size.

## Granularity on 600 Frames

This full-cluster experiment used twelve processes, dynamic scheduling,
600 frames, and CPU execution.

| Region grid | Tasks | Max compute time (s) | Total communication time (s) | Total idle time (s) | Idle-gap indicator | Balance result |
|---|---:|---:|---:|---:|---:|---|
| Two by two | 2400 | 14.981 | 121.984 | 95.091 | 0.718 | Not balanced |
| Four by three | 7200 | 47.291 | 362.261 | 329.268 | 0.759 | Not balanced |
| Five by four | 12000 | 96.959 | 777.383 | 744.691 | 0.837 | Not balanced |

Interpretation:

- Finer grids increased the number of tasks and increased total communication.
- Load balance still failed the 25 percent criterion for all tested grids.
- The five-by-four grid created more parallel work, but it also increased
  master-side coordination and result aggregation.
- A good report explanation is that object detection has irregular cost per
  tile and per frame, so granularity must be tuned rather than blindly increased.

## Earlier Partial-Cluster Results

When node2 was offline, the same larger-input experiments were also run on a
partial cluster using only master and node1. Those results are useful as
secondary evidence, but they should not replace the full three-machine numbers.

Partial static versus dynamic result:

| Scheduler | Runtime with communication (s) | Runtime without communication (s) | Load imbalance | Idle-gap indicator |
|---|---:|---:|---:|---:|
| Static | 114.082 | 108.339 | 1.593 | 0.733 |
| Dynamic | 170.670 | 162.645 | 2.680 | 0.833 |

Partial granularity result:

| Region grid | Tasks | Runtime with communication (s) | Runtime without communication (s) | Idle-gap indicator | Balance result |
|---|---:|---:|---:|---:|---|
| Two by two | 2400 | 41.880 | 35.506 | 0.826 | Not balanced |
| Four by three | 7200 | 113.257 | 106.502 | 0.837 | Not balanced |
| Five by four | 12000 | 170.670 | 162.645 | 0.833 | Not balanced |

## Speedup Repeats

Repeated speedup measurements on the 1200-frame workload were completed by
combining the original report sweep with a new repeat after node2 rejoined the
LAN. Both sweeps used CPU execution, a five-by-four region grid, dynamic
scheduling, and the process list one, two, four, eight, and twelve.

The new repeat is stored at:

```text
results/extra_report_live_20260623-185430/speedup_2N_repeat_1/
```

The two-repeat summary is stored at:

```text
results/extra_report_live_20260623-185430/speedup_2N_repeats_summary.csv
```

| Processes | Repeats | Mean runtime with communication (s) | Runtime stdev (s) | Mean speedup with communication |
|---:|---:|---:|---:|---:|
| 1 | 2 | 317.314 | 40.111 | 1.000 |
| 2 | 2 | 297.950 | 30.823 | 1.078 |
| 4 | 2 | 271.548 | 37.904 | 1.190 |
| 8 | 2 | 216.715 | 22.549 | 1.482 |
| 12 | 2 | 178.174 | 0.186 | 1.781 |

Interpretation:

- The twelve-process runtime is very stable across the two complete sweeps.
- The speedup value varies because the one-process baseline varies strongly
  between runs.
- This supports a careful report statement: the cluster throughput at twelve
  processes is stable, but speedup is sensitive to the serial baseline on a
  thermally and interactively used laptop cluster.

If the group gets another stable cluster window, the prepared command is:

```text
bash scripts/run/extra_report_experiments.sh
```

An additional third repeat would be useful, but it is not mandatory for the core
defense because the current results already include two complete 1200-frame
sweeps and the additional 600-frame three-machine sweep above.

## Recommendation for the Final Defense

Use these extra results as supporting discussion. The best defense framing is:

- The physical cluster helps most when the run actually spreads across multiple
  machines.
- Static scheduling outperformed dynamic scheduling for this tiled YOLO
  workload because the dynamic dispatcher added overhead.
- Load balance did not meet the 25 percent criterion, and the group measured
  that explicitly rather than hiding it.
- The project remains valid for the course because the report contains the
  parallel decomposition, mapping, communication, load-balance analysis,
  correctness checks, and speedup measurements required by the instructor.
