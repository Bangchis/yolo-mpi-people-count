# C++ MPI Source Map

`src/yolo_mpi_cpp.cpp` includes these files in report-friendly order. The code
is intentionally kept as one translation unit so the build stays simple:

```text
core/        config, task structs, task generation
detector/    mock detector, YOLO worker process, payload CSV text format
mpi/         static scheduler and MPI send/receive helpers
postprocess/ geometry, duplicate rules, temporal de-dup, frame merge
output/      CSV output, summary, correctness check
live/        camera tile input, live distribution, viewer output
```

Suggested reading order:

1. `core/types.hpp`
2. `core/tasks.hpp`
3. `mpi/static_scheduler.hpp`
4. `detector/yolo_worker_process.hpp`
5. `postprocess/duplicate_rules.hpp`
6. `output/csv.hpp`
7. `live/runner.hpp`

Most files are below 200 lines. The goal is not to hide complexity; it is to
make each course concept easy to find and explain.
