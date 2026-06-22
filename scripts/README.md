# Scripts Map

This folder has many scripts because the project needs cluster setup, live demo,
offline benchmarks, plots, and asset handling. Only a few scripts are part of
the core algorithm path.

## Core Runtime Helpers

These are called by the C++ MPI program or live pipeline.

- `yolo_worker.py`: local YOLO inference worker for one MPI rank.
- `camera_tile_source.py`: master camera/video reader for live mode.
- `live_viewer.py`: optional OpenCV live display on the master.
- `render_demo_video.py`: offline video rendering from C++ CSV output.

## Cluster Setup

These scripts prepare the three MacBooks.

- `collect_macos_node_info.sh`
- `write_macos_ssh_config.sh`
- `check_cluster_macos.sh`
- `sync_to_nodes.sh`
- `setup_yolo_macos.sh`
- `import_cluster_reference.sh`
- `summarize_host_slots.py`

## Main Run Entrypoints

These are the commands students usually run.

- `build.sh`: build the C++17/OpenMPI binary.
- `run_live_camera_demo.sh`: live camera demo from the master camera.
- `run_demo_correctness.sh`: serial-vs-MPI correctness smoke.
- `run_demo_perf.sh`: one offline performance run.
- `run_report_mot17_mini.sh`: full report workflow on the small MOT17 asset.
- `run_report_mot17_fullseq.sh`: full report workflow on the 600-frame asset.
- `run_mot17_fullseq_accuracy_suite.sh`: accuracy across multiple full sequences.

## Report Metrics

These scripts generate the CSV files and plots needed by the report rubric.

- `compare_frame_counts.py`: serial vs MPI frame-count comparison.
- `evaluate_count_accuracy.py`: predicted count vs MOT17 ground truth.
- `run_find_N.sh`: runtime vs input size.
- `run_speedup_sweep.sh`: speedup for P=1,2,4,8,12.
- `run_required_experiments.sh`: older compact experiment runner kept for backup.

## Assets And Dataset

These scripts avoid committing heavy files to GitHub.

- `download_model.py`
- `upload_hf_assets.py`
- `download_hf_assets.py`
- `prepare_mot17_mini.py`
- `probe_video.py`

## Read This First

If you need to explain the project, read in this order:

1. `docs/03_PARALLEL_ALGORITHM_SHORT.md`
2. `docs/02_CODE_READING_GUIDE.md`
3. `src/yolo_mpi_cpp.cpp`
4. `src/yolo_mpi/mpi_scheduling.hpp`
5. `scripts/yolo_worker.py`

Do not delete a script just because it is not in the live demo path. Some
scripts exist only to produce report evidence: find N, speedup, load balance,
accuracy, or cluster proof.
