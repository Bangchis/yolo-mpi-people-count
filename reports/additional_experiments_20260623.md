# Additional Experiments on 23 June 2026

This note records extra experiments requested after the main report had already
been prepared. These results are useful for discussion and defense, but they
should not replace the official three-machine benchmark tables in the final
report because node2 was unreachable during this rerun.

## Cluster Availability During Rerun

The master and node1 were reachable, but node2 was not reachable.

Observed status:

- master SSH: reachable
- node1 SSH: reachable
- node2 SSH: failed, port 22 timed out
- node2 ping: 100 percent packet loss

Because of this, new full three-machine experiments could not be rerun during
this session. The official three-machine measurements already present in the
main report remain the primary results.

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

For comparison, the official three-machine result in the main report processed
the same 600-frame scale with twelve processes in 123.667 seconds. This shows
that the physical cluster can provide a substantial advantage over the local
master-only run, although the comparison should be presented carefully because
the runs were not repeated under identical thermal and network conditions.

## Static versus Dynamic Scheduling on a Larger Input

Because node2 was unavailable, this experiment was run on a partial cluster
using master and node1 only. It used eight processes, 600 frames, a five-by-four
region grid, and CPU execution.

| Scheduler | Runtime with communication (s) | Runtime without communication (s) | Load imbalance | Idle-gap indicator |
|---|---:|---:|---:|---:|
| Static | 114.082 | 108.339 | 1.593 | 0.733 |
| Dynamic | 170.670 | 162.645 | 2.680 | 0.833 |

Interpretation:

- Static scheduling remained faster on this workload.
- Dynamic scheduling had much higher communication and idle time.
- The result strengthens the conclusion that dynamic scheduling is more
  flexible but not automatically faster.
- For this YOLO workload, the per-task dispatch overhead is significant.

## Granularity on a Larger Input

This partial-cluster experiment used master and node1, eight processes,
600 frames, dynamic scheduling, and CPU execution.

| Region grid | Tasks | Runtime with communication (s) | Runtime without communication (s) | Idle-gap indicator | Balance result |
|---|---:|---:|---:|---:|---|
| Two by two | 2400 | 41.880 | 35.506 | 0.826 | Not balanced |
| Four by three | 7200 | 113.257 | 106.502 | 0.837 | Not balanced |
| Five by four | 12000 | 170.670 | 162.645 | 0.833 | Not balanced |

Interpretation:

- Finer region grids increased the number of tasks and the amount of work.
- Idle imbalance remained above the 25 percent criterion.
- Increasing input length alone did not fully solve load balance.
- Communication and master-side coordination still dominated a large part of
  the runtime.

## Speedup Repeats

Repeated speedup measurements were not run in this session because node2 was
offline and the local-only baseline was already very time-consuming. When all
three machines are online again, the recommended next step is to rerun the
official speedup sweep for two or three repetitions and report the mean runtime.

## Recommendation for the Final Defense

Use these extra results as supporting discussion, not as the main benchmark.
The best defense framing is:

- The official report already satisfies the required experiments.
- Additional local-only testing shows that a single machine does not scale
  reliably for this tiled YOLO workload.
- Larger-input scheduling tests show that dynamic scheduling has real overhead
  and should be justified as a load-balancing strategy, not claimed as always
  faster.
- The load-balance issue is real and measured, not ignored.
