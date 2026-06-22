# Huong Dan Doc Code Va Chia Viec

Muc tieu cua tai lieu nay la giup ca nhom doc code theo dung luong chay, khong
bi lac trong cac script phu. Du an co hai lop:

1. Lop thuat toan song song: C++17 + OpenMPI.
2. Lop ho tro: Python/shell cho YOLO worker, camera, ve bieu do, tai asset.

Python khong chay MPI. MPI chi nam trong C++.

## 1. Luong Chay Chinh

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

## 2. C++ Runtime Files

`src/yolo_mpi_cpp.cpp`

- Entrypoint ngan.
- Goi `MPI_Init`, doc config, chon live/offline, ghi ket qua cuoi.

`src/yolo_mpi/common.hpp`

- Khai bao `Config`, `Task`, `Detection`, `Metrics`, `ImageTask`.
- Day la cac kieu du lieu can hieu truoc khi doc cac file khac.

`src/yolo_mpi/config_and_tasks.hpp`

- Parse command-line options.
- Tao task tu video: moi task la mot tile cua mot frame.
- Co mock detector va command detector dung cho test nhanh.

`src/yolo_mpi/detector_worker.hpp`

- Quan ly tien trinh Python `scripts/runtime/yolo_worker.py`.
- Moi MPI rank giu worker song de model YOLO chi load mot lan.
- Chuyen task sang worker va nhan bbox tra ve.

`src/yolo_mpi/mpi_scheduling.hpp`

- Static scheduling: block-cyclic mapping.
- Dynamic scheduling: rank 0 lam master queue, worker xin viec tiep.
- Do `comm_ms` quanh `MPI_Send`, `MPI_Recv`, `MPI_Gather/Gatherv`.

`src/yolo_mpi/postprocess_output.hpp`

- Tile-owner filter.
- Global NMS/de-dup.
- Ghi `frame_counts.csv`, `bboxes.csv`, `rank_metrics.csv`, `summary.csv`.

`src/yolo_mpi/live_pipeline.hpp`

- Live camera pipeline.
- Rank 0 doc camera, cat tile, gui tile cho worker, nhan ket qua va hien live.

## 3. Python Files Quan Trong

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

## 4. Shell Scripts Nen Nho

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

## 5. De Xuat Chia Viec 4 Nguoi

Nguoi 1: C++ MPI scheduling

- Doc `mpi_scheduling.hpp`.
- Giai thich static/dynamic scheduling, communication, load balance.
- Phu trach bang speedup va granularity.

Nguoi 2: YOLO worker va post-processing

- Doc `detector_worker.hpp`, `postprocess_output.hpp`, `yolo_worker.py`.
- Giai thich bbox remap, tile-owner filter, NMS/de-dup.
- Phu trach correctness serial vs MPI.

Nguoi 3: Cluster va asset/data

- Doc scripts SSH/sync/setup/download/upload.
- Phu trach ket noi 3 Mac, hostfile, evidence, Hugging Face asset.

Nguoi 4: Report va plots

- Doc `scripts/run/report_mot17_mini.sh`, `scripts/run/report_mot17_fullseq.sh`, `scripts/report/plots/*.py`.
- Phu trach bang bieu, accuracy voi MOT17 ground truth, tong hop ket qua.

## 6. File Khong Can Hoc Sau

Nhung file nay la tool ho tro, chi can biet cong dung:

- `scripts/assets/upload_hf_assets.py`
- `scripts/assets/download_hf_assets.py`
- `scripts/assets/prepare_mot17_mini.py`
- `scripts/runtime/render_demo_video.py`
- `scripts/report/check_final_readiness.py`

Khong nen xoa chung vi chung giup setup lai may moi va tao bang bieu report.
