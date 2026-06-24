# Scripts Map

This folder has many scripts because the project needs cluster setup, live demo,
offline benchmarks, plots, and asset handling. Only a few scripts are part of
the core algorithm path.

## Core Runtime Helpers: `scripts/runtime/`

These are called by the C++ MPI program or live pipeline.

- `scripts/runtime/yolo_worker.py`: local YOLO inference worker for one MPI rank.
- `scripts/runtime/camera_tile_source.py`: master camera/video reader for live mode.
- `scripts/runtime/live_viewer.py`: optional OpenCV live display on the master.
- `scripts/runtime/render_demo_video.py`: offline video rendering from C++ CSV output.

## Cluster Setup: `scripts/cluster/`

These scripts prepare the three MacBooks.

- `scripts/cluster/collect_node_info.sh`
- `scripts/cluster/write_ssh_config.sh`
- `scripts/cluster/check_macos.sh`
- `scripts/cluster/sync_to_nodes.sh`
- `scripts/cluster/setup_yolo_macos.sh`
- `scripts/cluster/import_reference.sh`
- `scripts/cluster/summarize_host_slots.py`

## Main Run Entrypoints: `scripts/run/`

These are the commands students usually run.

- `scripts/build.sh`: build the C++17/OpenMPI binary.
- `scripts/run/live_camera_demo.sh`: live camera demo from the master camera.
- `scripts/run/demo_correctness.sh`: serial-vs-MPI correctness smoke.
- `scripts/run/demo_perf.sh`: one offline performance run.
- `scripts/run/report_mot17_mini.sh`: full report workflow on the small MOT17 asset.
- `scripts/run/report_mot17_fullseq.sh`: full report workflow on the 600-frame asset.
- `scripts/run/mot17_fullseq_accuracy_suite.sh`: accuracy across multiple full sequences.

## Report Metrics: `scripts/report/`

These scripts generate the CSV files and plots needed by the report rubric.

- `scripts/report/compare_frame_counts.py`: serial vs MPI frame-count comparison.
- `scripts/report/evaluate_count_accuracy.py`: predicted count vs MOT17 ground truth.
- `scripts/run/find_N.sh`: runtime vs input size.
- `scripts/run/speedup_sweep.sh`: speedup for P=1,2,4,8,12.

## Assets And Dataset: `scripts/assets/`

These scripts avoid committing heavy files to GitHub.

- `scripts/assets/download_model.py`
- `scripts/assets/upload_hf_assets.py`
- `scripts/assets/download_hf_assets.py`
- `scripts/assets/prepare_mot17_mini.py`
- `scripts/assets/probe_video.py`

## Read This First

If you need to explain the project, read in this order:

1. `docs/03_PARALLEL_ALGORITHM_SHORT.md`
2. `docs/02_CODE_READING_GUIDE.md`
3. `src/yolo_mpi_cpp.cpp`
4. `scripts/runtime/yolo_worker.py`

Do not delete a script just because it is not in the live demo path. Some
scripts exist only to produce report evidence: find N, speedup, load balance,
accuracy, or cluster proof.
