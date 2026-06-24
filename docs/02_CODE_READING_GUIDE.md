# Huong Dan Doc Code Va Chia Viec

Muc tieu cua tai lieu nay la giup ca nhom doc code theo dung luong chay, khong
bi lac trong cac script phu. Du an co hai method chinh:

1. Method 1: YOLO11n video inference, song song hoa tac vu.
2. Method 2: VGG11 no-BN convolution, song song hoa du lieu trong layer.

Ca hai method deu dung C++17 + OpenMPI. Python khong chay MPI. Python chi ho
tro YOLO worker, camera, ve bieu do, tai asset va tao report.

## 1. Luong Chay Method 1

```text
mpirun
  -> build/yolo_mpi_cpp
    -> tao danh sach task frame/tile
    -> chia task cho MPI rank
    -> moi rank goi scripts/runtime/yolo_worker.py neu detector=yolo
    -> rank 0 gom bbox
    -> rank 0 loc trung/NMS
    -> ghi CSV ket qua
```

Neu chay live camera:

```text
rank 0 doc camera
  -> cat frame thanh tile JPEG
  -> gui tile cho cac rank
  -> cac rank detect
  -> rank 0 gom bbox va hien cua so live
```

## 2. Luong Chay Method 2

```text
mpirun
  -> build/vgg11_mpi
    -> tao input tensor deterministic
    -> chay VGG11 no-BN convolution stack
    -> moi Conv2D chia feature map thanh block 2D
    -> cac rank trao doi halo voi hang xom
    -> moi rank tinh local convolution
    -> rank 0 gather output full tensor
    -> so sanh output distributed voi serial baseline
    -> ghi summary/layer/rank/topology metrics
```

Method 2 co hai communication modes:

```text
blocking    -> MPI_Sendrecv
nonblocking -> MPI_Irecv + MPI_Isend + MPI_Waitall
```

Method 2 con ghi topology-aware mapping:

```text
topology_mapping.csv  -> rank nao nam o cell nao trong grid 2D
topology_metrics.csv  -> bao nhieu halo edges intra-machine/inter-machine
topology_grid.txt     -> luoi rank de nhin nhanh khi thuyet trinh
```

## 3. C++ Runtime Files

`src/yolo_mpi_cpp.cpp`

- Entrypoint ngan.
- Goi `MPI_Init`, doc config, chon live/offline, ghi ket qua cuoi.

`src/yolo_mpi/core/types.hpp`

- Khai bao `Config`, `Task`, `Detection`, `Metrics`, `ImageTask`.
- Day la cac kieu du lieu can hieu truoc khi doc cac file khac.

`src/yolo_mpi/core/config.hpp` va `src/yolo_mpi/core/tasks.hpp`

- Parse command-line options.
- Tao task tu video: moi task la mot tile cua mot frame.

`src/yolo_mpi/detector/`

- Quan ly tien trinh Python `scripts/runtime/yolo_worker.py`.
- Moi MPI rank giu worker song de model YOLO chi load mot lan.
- Chuyen task sang worker va nhan bbox tra ve.

`src/yolo_mpi/mpi/`

- Static scheduling: block-cyclic mapping.
- Dynamic scheduling: rank 0 lam master queue, worker xin viec tiep.
- Do `comm_ms` quanh `MPI_Send`, `MPI_Recv`, `MPI_Gather/Gatherv`.

`src/yolo_mpi/postprocess/` va `src/yolo_mpi/output/`

- Tile-owner filter.
- Global NMS/de-dup.
- Ghi `frame_counts.csv`, `bboxes.csv`, `rank_metrics.csv`, `summary.csv`.

`src/yolo_mpi/live/`

- Live camera pipeline.
- Rank 0 doc camera, cat tile, gui tile cho worker, nhan ket qua va hien live.

`src/vgg11_mpi.cpp`

- Entrypoint cho Method 2.
- Parse tham so `--height`, `--width`, `--profile`, `--grid`, `--halo-mode`.
- Chay serial baseline va distributed VGG11 convolution stack.
- Ghi `summary.csv`, `layer_metrics.csv`, `rank_metrics.csv`, topology files.

`src/vgg11_mpi/core/tensor.hpp`

- Tensor C x H x W don gian.
- Tao input/weights deterministic.
- Serial Conv2D, ReLU, MaxPool, compare tensor error.

`src/vgg11_mpi/core/partition.hpp`

- Chia feature map thanh block 2D.
- Dung cho mapping rank -> block.

`src/vgg11_mpi/conv/`

- Scatter core block.
- Halo exchange blocking/nonblocking.
- Local Conv2D tren block co halo.
- Gather output block ve rank 0.

`src/vgg11_mpi/runner/` va `src/vgg11_mpi/output/`

- Chay serial/distributed VGG11 stack.
- Gom rank metrics.
- Ghi CSV va topology files.

## 4. Python Files Quan Trong

`scripts/runtime/yolo_worker.py`

- Worker inference cuc bo cho moi rank.
- Nhan lenh `TASK` hoac `IMAGE` qua stdin.
- Tra bbox ve stdout.

`scripts/runtime/camera_tile_source.py`

- Chi dung cho live camera.
- Doc camera/video tren master, cat tile JPEG.

`scripts/runtime/live_viewer.py`

- Hien cua so OpenCV tren master.

`scripts/report/evaluate_count_accuracy.py`

- So sanh YOLO count voi MOT17 ground truth count.

`scripts/report/compare_frame_counts.py`

- So sanh serial vs MPI de chung minh song song khong doi ket qua.

`scripts/report/plots/*.py`

- Ve hinh cho report: find N, speedup, rank metrics, count error.
- Method 2 plots: `plot_vgg11_conv.py`, `plot_vgg11_input_size.py`.

## 5. Shell Scripts Nen Nho

`scripts/build.sh`

- Build C++ binary.

`scripts/cluster/check_macos.sh`

- Check SSH/MPI/hostfile tren 3 may.

`scripts/cluster/sync_to_nodes.sh`

- Sync repo, model, data sang node1/node2.

`scripts/run/report_mot17_mini.sh`

- Chay day du thi nghiem nho de lay so lieu report nhanh.

`scripts/run/report_mot17_fullseq.sh`

- Chay report voi MOT17 full sequence chinh.

`scripts/run/mot17_fullseq_accuracy_suite.sh`

- Chay accuracy tren nhieu full sequence: MOT17-02, 05, 09, 10.

`scripts/run/vgg11_report_experiments.sh`

- Chay report suite cho Method 2.
- Gom input-size, speedup/efficiency, blocking vs nonblocking, topology mapping.

`scripts/run/vgg11_conv_benchmark.sh`

- Chay speedup/communication benchmark rieng cho Method 2.

`scripts/run/vgg11_topology_mapping_comparison.sh`

- So sanh topology-aware placement voi round-robin placement tren 3 may.

## 6. De Xuat Chia Viec 4 Nguoi

Nguoi 1: C++ MPI scheduling

- Doc `src/yolo_mpi/mpi/static_scheduler.hpp` va `src/yolo_mpi/mpi/dynamic_scheduler.hpp`.
- Giai thich static/dynamic scheduling, communication, load balance.
- Phu trach bang speedup va granularity.

Nguoi 2: YOLO worker va post-processing

- Doc `src/yolo_mpi/detector/`, `src/yolo_mpi/postprocess/`, `scripts/runtime/yolo_worker.py`.
- Giai thich bbox remap, tile-owner filter, NMS/de-dup.
- Phu trach correctness serial vs MPI.

Nguoi 3: Cluster va asset/data

- Doc scripts SSH/sync/setup/download/upload.
- Phu trach ket noi 3 Mac, hostfile, evidence, Hugging Face asset.

Nguoi 4: Report va plots

- Doc `scripts/run/report_mot17_mini.sh`, `scripts/run/report_mot17_fullseq.sh`,
  `scripts/run/vgg11_report_experiments.sh`, `scripts/report/plots/*.py`.
- Phu trach bang bieu, accuracy voi MOT17 ground truth, tong hop ket qua.

Neu muon chia Method 2 rieng:

Nguoi 1 doc `src/vgg11_mpi/conv/halo_exchange.hpp` va `src/vgg11_mpi/conv/distributed_conv.hpp` de giai thich halo exchange.
Nguoi 2 doc `src/vgg11_mpi/runner/vgg11_runner.hpp` va `src/vgg11_mpi.cpp` de giai thich VGG11 stack va correctness.
Nguoi 3 chay `scripts/run/vgg11_report_experiments.sh` tren 3 may.
Nguoi 4 dua hinh/bang Method 2 vao report.

## 7. File Khong Can Hoc Sau

Nhung file nay la tool ho tro, chi can biet cong dung:

- `scripts/assets/upload_hf_assets.py`
- `scripts/assets/download_hf_assets.py`
- `scripts/assets/prepare_mot17_mini.py`
- `scripts/runtime/render_demo_video.py`
- `scripts/report/check_final_readiness.py`

Khong nen xoa chung vi chung giup setup lai may moi va tao bang bieu report.
