# BÁO CÁO BÀI TẬP LỚN

## Đề tài: Song song hóa bài toán đếm người trong video bằng YOLO và C++17/OpenMPI trên cụm 3 máy MacBook

**Môn học:** Parallel Programming and Computing  
**Nhóm:** `<ID nhóm>`  
**Sinh viên:** `<Danh sách thành viên>`  
**Ngày nộp:** 24/06/2026  

---

## Tóm Tắt

Bài toán của nhóm là đếm số người xuất hiện trong từng khung hình video bằng mô hình YOLO đã huấn luyện sẵn. Khối lượng tính toán chính nằm ở bước suy luận mô hình trên nhiều frame và nhiều vùng ảnh. Nhóm triển khai chương trình song song bằng **C++17/OpenMPI** trên cụm gồm ba máy vật lý trong cùng mạng LAN. Mỗi tiến trình MPI nhận một phần công việc, gọi worker YOLO cục bộ để suy luận, sau đó gửi bounding box và số liệu thời gian về tiến trình gốc để hợp nhất kết quả.

Thuật toán dùng **song song cấp tác vụ** với **phân rã lai temporal-spatial**: video được tách theo thời gian thành các frame, mỗi frame được tách tiếp theo không gian thành các tile. Một tác vụ là một cặp `(frame_id, tile_id)` kèm tọa độ vùng ảnh. Nhóm hỗ trợ hai chiến lược lập lịch: static block-cyclic và dynamic master-worker. Trong báo cáo, dynamic scheduling là cấu hình chính vì thời gian YOLO trên từng tile không đồng đều.

Kết quả cần chứng minh gồm: đúng/sai giữa bản serial và MPI, sai số đếm so với ground truth MOT17, tìm kích thước đầu vào N cho thời gian chạy 2-3 phút, đánh giá granularity/load balance, và đo speedup khi thay đổi số tiến trình.

---

## Mục Lục

1. Giới thiệu bài toán  
2. Cơ sở lý thuyết và công nghệ sử dụng  
3. Phân tích song song hóa  
4. Thiết kế thuật toán song song  
5. Cài đặt hệ thống MPI cluster  
6. Thiết kế thực nghiệm  
7. Kết quả thực nghiệm  
8. Nhận xét, hạn chế và hướng phát triển  
9. Kết luận  
10. Tài liệu tham khảo  

---

## 1. Giới Thiệu Bài Toán

### 1.1. Bối cảnh

Đếm người trong video là bài toán thực tế trong giám sát lớp học, phân tích mật độ đám đông, thống kê lưu lượng người đi bộ và quản lý không gian công cộng. Nếu chạy tuần tự từng frame, thời gian xử lý tăng gần tuyến tính theo số lượng frame. Với video dài hoặc độ phân giải cao, chương trình tuần tự khó đáp ứng yêu cầu thời gian.

Do mô hình YOLO đã được huấn luyện sẵn, trọng tâm của đề tài không phải huấn luyện deep learning mà là **song song hóa pipeline suy luận video**. Đây là bài toán phù hợp với MPI vì các frame/tile có thể xử lý độc lập, sau đó gom kết quả ở cuối.

### 1.2. Phát biểu bài toán

**Input:**

- Video đầu vào, gồm `N` frame.
- Mô hình YOLO pretrained.
- Các tham số: độ tin cậy `conf`, ngưỡng NMS `iou`, lưới tile `C x R`, số tiến trình MPI `P`.

**Output:**

- Số người trên từng frame.
- Bounding box sau hậu xử lý.
- Thời gian chạy toàn chương trình.
- Thời gian tính toán, giao tiếp và nhàn rỗi theo từng tiến trình.

**Mục tiêu song song:**

- Giảm thời gian xử lý video bằng cách chia task cho nhiều tiến trình MPI.
- Kiểm tra kết quả MPI khớp với baseline serial.
- Đánh giá cân bằng tải, độ mịn task và speedup.

---

## 2. Cơ Sở Lý Thuyết Và Công Nghệ Sử Dụng

### 2.1. YOLO cho phát hiện người

YOLO là họ mô hình object detection một giai đoạn, trả về bounding box, class và confidence trong một lần suy luận. Trong đề tài, nhóm dùng YOLO pretrained và chỉ lọc class `person`. Với mỗi tile hoặc frame, YOLO trả về danh sách bounding box cục bộ. Các bounding box này được remap về tọa độ frame gốc trước khi hợp nhất.

### 2.2. OpenMPI

OpenMPI tạo nhiều tiến trình độc lập gọi là MPI rank. Mỗi rank chạy cùng chương trình C++ nhưng có `rank` khác nhau. Các rank giao tiếp qua `MPI_Send`, `MPI_Recv`, `MPI_Gather` và `MPI_Gatherv`.

Trong đề tài:

```text
MPI rank -> C++ process -> local YOLO worker -> bbox payload
```

MPI không song song hóa bên trong mô hình YOLO. MPI song song hóa ở cấp pipeline/tác vụ: nhiều rank cùng xử lý nhiều frame/tile khác nhau.

### 2.3. Dataset MOT17

Nhóm sử dụng dữ liệu MOT17 làm tập kiểm thử vì đây là bộ dữ liệu phổ biến cho bài toán người đi bộ trong video. Để phục vụ benchmark trong thời gian môn học, nhóm chuẩn bị:

- `MOT17-mini`: 300 frame đầu của một sequence, dùng cho smoke test và thực nghiệm nhanh.
- `MOT17-fullseq`: một số sequence đầy đủ hơn, dùng để kiểm tra stress test và kết quả ổn định hơn.

Ground truth được chuyển thành file `*_counts.csv`, mỗi dòng gồm `frame_id` và số người thật trên frame.

---

## 3. Phân Tích Song Song Hóa

### 3.1. Song song cấp độ nào?

Thuật toán dùng **song song cấp tác vụ**.

Mỗi task là một công việc suy luận độc lập:

```text
task = (frame_id, tile_id, x1, y1, x2, y2)
```

Task có thể là toàn bộ frame hoặc một vùng tile trong frame. Các task không cần trao đổi dữ liệu trong lúc YOLO đang chạy. Giao tiếp chỉ xảy ra khi rank nhận task và gửi kết quả về rank 0.

### 3.2. Kỹ thuật phân rã

Nhóm sử dụng **hybrid decomposition**:

1. **Temporal decomposition:** chia video theo frame.
2. **Spatial decomposition:** chia mỗi frame thành lưới tile `C x R`.

Ví dụ:

```text
N = 300 frame
tile_grid = 4x3
tasks_per_frame = 12
total_tasks = 300 * 12 = 3600
```

Phân rã theo frame giúp tăng số lượng công việc khi video dài. Phân rã theo tile giúp tăng độ mịn khi số frame chưa đủ lớn hoặc khi muốn dùng nhiều process hơn.

### 3.3. Mapping technique

Lưới 2D tile được flatten thành danh sách task 1D:

```text
task_id = frame_id * (cols * rows) + tile_id
```

Vì vậy mapping chính là **1D task mapping** trên danh sách task đã flatten.

Với static scheduling:

```text
block_id = task_id / chunk_size
assigned_rank = block_id mod P
```

Với dynamic scheduling:

```text
rank 0 giữ queue task
rank nào rảnh thì nhận task kế tiếp
```

Dynamic scheduling phù hợp hơn vì mỗi tile có độ khó khác nhau. Tile chứa nhiều người hoặc nhiều vật thể dễ mất nhiều thời gian hơn tile trống.

### 3.4. Communication strategy and topology

Topology của chương trình là **master-worker/star topology**:

```text
rank 0: master/coordinator
rank 1..P-1: worker
```

Luồng giao tiếp dynamic:

```text
rank 0 -> worker: task metadata
worker -> rank 0: bbox + metrics payload
rank 0 -> worker: task mới hoặc stop signal
```

Chương trình dùng blocking communication ở các bước nhận/gửi chính. Rank 0 có thêm polling bằng `MPI_Iprobe` để vừa kiểm tra worker xong việc vừa tự tính toán khi `master_compute=1`.

Trong static mode, mỗi rank tự chọn task theo công thức block-cyclic, sau đó toàn bộ kết quả được gom bằng `MPI_Gather/Gatherv`.

### 3.5. Load balancing

Load balancing có hai mức:

- **Dynamic scheduling:** rank nào xử lý nhanh hơn sẽ nhận nhiều task hơn.
- **Granularity tuning:** thay đổi `tile_grid` và `chunk_size` để điều chỉnh độ mịn.

Nếu task quá thô, một rank có thể giữ task nặng quá lâu, các rank khác rảnh. Nếu task quá mịn, số lượng task tăng mạnh làm tăng chi phí giao tiếp và hậu xử lý. Do đó nhóm đo nhiều cấu hình tile grid để chọn mức phù hợp.

---

## 4. Thiết Kế Thuật Toán Song Song

### 4.1. Pipeline tổng quát

```text
Video input
  -> đọc frame
  -> chia tile
  -> tạo danh sách task
  -> MPI scheduling
  -> YOLO inference trên từng task
  -> remap bbox về frame gốc
  -> tile-owner filter
  -> global NMS / de-dup
  -> count people
  -> xuất CSV/plot/video
```

### 4.2. Hậu xử lý bounding box

Vì một người có thể nằm ở vùng overlap giữa hai tile, cùng một người có thể được phát hiện nhiều lần. Hậu xử lý gồm:

1. **Remap bbox:** bbox trong tile được cộng offset để trở về tọa độ frame gốc.
2. **Tile-owner filter:** chỉ giữ bbox ở tile mà tâm bbox thuộc về vùng core của tile đó.
3. **Global NMS:** so sánh bbox toàn frame bằng IoU/IoS để loại trùng.
4. **Optional de-dup rules:** xử lý trường hợp người sát camera bị cắt qua biên tile.

Trong live camera, nhóm bổ sung chế độ anchor full-frame để tránh tile false positive làm đếm thừa. Tuy nhiên benchmark/report chính vẫn dùng offline video để đo MPI rõ ràng.

### 4.3. Mã giả thuật toán dynamic scheduling

```text
Input: tasks[0..T-1], P MPI ranks
Output: detections, rank_metrics

if rank == 0:
    next = 0
    active = 0

    for worker in 1..P-1:
        if next < T:
            send tasks[next] to worker
            next += 1
            active += 1
        else:
            send STOP to worker

    while completed < T:
        while worker_result_available():
            payload = receive result from any worker
            append payload to global result
            completed += 1
            active -= 1

            if next < T:
                send tasks[next] to that worker
                next += 1
                active += 1
            else:
                send STOP to that worker

        if master_compute and next < T:
            process tasks[next] locally on rank 0
            completed += 1
            next += 1
        else:
            block until one worker result arrives

    postprocess all detections
    write CSV files

else:
    create YOLO detector once
    loop:
        task = receive from rank 0
        if task is STOP:
            break
        detections = YOLO(task)
        send detections and metrics to rank 0
```

### 4.4. Mã giả hậu xử lý một frame

```text
Input: detections from all ranks for one frame
Output: final detections

filtered = []
for det in detections:
    if center(det) belongs to owner tile core:
        filtered.append(det)

sort filtered by confidence descending
kept = []

for candidate in filtered:
    duplicate = false
    for selected in kept:
        if IoU(candidate, selected) >= threshold:
            duplicate = true
        if IoS(candidate, selected) >= threshold:
            duplicate = true
    if not duplicate:
        kept.append(candidate)

return kept
```

---

## 5. Cài Đặt Hệ Thống MPI Cluster

### 5.1. Cấu hình phần cứng

Cụm gồm ba máy vật lý trong cùng mạng LAN:

| Role | Host alias | Vai trò | Ghi chú |
|---|---|---|---|
| master | `master` | Điều phối, chạy rank 0, lưu kết quả | Có camera cho demo live |
| node1 | `node1` | Worker | Máy mạnh hơn, có thể nhận nhiều slot |
| node2 | `node2` | Worker | Worker thứ hai |

Chương trình không sử dụng cloud server. Các máy bật SSH/Remote Login và OpenMPI. Repo được đồng bộ sang node bằng `rsync`.

### 5.2. Cấu hình hostfile

Benchmark CPU theo yêu cầu môn học dùng hostfile core:

```text
master slots=4
node1 slots=4
node2 slots=4
```

Demo live CPU max dùng hostfile weighted:

```text
localhost slots=10
node1 slots=12
node2 slots=8
```

Benchmark report chính ưu tiên `configs/hosts_macos_core` để giải thích dễ theo số process/core. Live camera dùng cấu hình riêng và không trộn vào kết quả benchmark chính.

### 5.3. MPI flags

Các lệnh MPI dùng TCP qua LAN:

```bash
--mca btl tcp,self
--mca btl_tcp_if_include 172.1.0.0/24
--mca btl_tcp_disable_family 6
--hetero-nodes
```

`--hetero-nodes` cần thiết vì ba máy Mac có cấu hình khác nhau.

---

## 6. Thiết Kế Thực Nghiệm

### 6.1. Dataset

Dataset chính cho báo cáo:

```text
data/mot17-mini/MOT17-02-SDP-300_960x540.mp4
data/mot17-mini/MOT17-02-SDP-300_counts.csv
```

Dataset phụ/stress test:

```text
data/mot17-fullseq/
```

Lý do chọn MOT17:

- Là dataset phổ biến cho bài toán người trong video.
- Có scene đám đông và occlusion nên khó hơn ảnh demo đơn giản.
- Có ground truth để đánh giá sai số đếm.

### 6.2. Baseline

Nhóm dùng hai baseline:

1. **Serial YOLO baseline:** `P=1`, cùng model, cùng threshold, cùng tile grid.
2. **Ground truth count:** số người thật từ MOT17 để đo MAE/RMSE.

Serial vs MPI kiểm tra tính đúng của song song hóa. YOLO vs ground truth kiểm tra chất lượng ứng dụng đếm người.

### 6.3. Các thí nghiệm theo yêu cầu thầy

| Mục yêu cầu | Cách đo | File kết quả |
|---|---|---|
| Correctness serial vs MPI | So sánh `frame_counts.csv` | `correctness/correctness_compare.csv` |
| Accuracy vs ground truth | MAE, RMSE, exact match | `accuracy/accuracy.csv` |
| Tìm N 2-3 phút | Chạy nhiều frame count | `find_N/raw/find_N.csv` |
| Granularity/load balance | Chạy nhiều tile grid | `granularity/granularity_overview.csv` |
| Stacked rank time | compute + comm + idle | `rank_metrics_stacked.png` |
| Speedup | P = 1,2,4,8,12 | `speedup/raw/speedup.csv`, `speedup.png` |

### 6.4. Lệnh chạy full report

```bash
cd "/Users/bangbang/Desktop/code python/yolo-mpi-people-count"

bash scripts/cluster/write_ssh_config.sh
bash scripts/cluster/sync_to_nodes.sh
YOLO_SETUP_REMOTE=1 bash scripts/cluster/setup_yolo_macos.sh
bash scripts/cluster/check_macos.sh

YOLO_REPORT_DIR="results/report_mot17_mini_final_$(date +%Y%m%d-%H%M%S)" \
YOLO_DEVICE=cpu \
YOLO_IMGSZ=320 \
YOLO_TILE_GRID=4x3 \
YOLO_SCHEDULE=dynamic \
YOLO_MASTER_COMPUTE=1 \
YOLO_REPORT_MPI_NP=12 \
YOLO_REPORT_HOSTFILE=configs/hosts_macos_core \
YOLO_FIND_FRAME_LIST="30 60 100 150 220 300" \
YOLO_GRANULARITY_GRIDS="1x1 2x2 4x3 5x4" \
YOLO_P_LIST="1 2 4 8 12" \
YOLO_SPEEDUP_FRAMES=300 \
bash scripts/run/report_mot17_mini.sh
```

Sau khi chạy:

```bash
.venv/bin/python scripts/report/summarize_report_dir.py \
  --report-dir results/report_mot17_mini_final_<timestamp>
```

Script này tạo `summary_tables.md` để copy bảng vào báo cáo.

---

## 7. Kết Quả Thực Nghiệm

> Ghi chú: các bảng dưới đây cần thay bằng kết quả full run cuối cùng trước khi nộp. Bản quick run đã được chạy để xác nhận pipeline hoạt động, nhưng không dùng làm kết quả chính vì số frame quá ít.

### 7.1. Kiểm tra correctness: serial vs MPI

Mục tiêu là chứng minh song song hóa không làm sai kết quả so với cách chạy tuần tự cùng cấu hình.

| Thí nghiệm | Kết quả mong muốn |
|---|---|
| Serial vs MPI | `correctness_pass = YES` |
| Sai số max | `0` |
| Sai số trung bình | `0` |

Kết quả full run lấy từ:

```text
correctness/correctness_compare.csv
correctness/correctness_per_frame.csv
```

### 7.2. Đánh giá YOLO count so với ground truth

Metric:

```text
MAE  = mean(abs(pred_count - gt_count))
RMSE = sqrt(mean((pred_count - gt_count)^2))
Exact match rate = tỷ lệ frame đếm đúng tuyệt đối
```

Kết quả lấy từ:

```text
accuracy/accuracy.csv
accuracy/per_frame_accuracy.csv
accuracy/count_error_plot.png
```

Cần giải thích rõ: nếu YOLO pretrained đếm lệch so với ground truth, đó là sai số mô hình nhận diện, không phải sai số song song hóa. Song song hóa đúng được kiểm chứng bằng serial-vs-MPI.

### 7.3. Xác định kích thước dữ liệu N

Theo yêu cầu, chọn số process bằng số core/process chính của cụm, ví dụ `P=12`. Chạy nhiều kích thước frame để tìm N sao cho runtime khoảng 2-3 phút.

Kết quả lấy từ:

```text
find_N/raw/find_N.csv
find_N/figures/find_N_runtime.png
```

Bảng cần điền sau full run:

| N frames | Time with comm (s) | Time without comm (s) | Nhận xét |
|---:|---:|---:|---|
| 30 | `<fill>` | `<fill>` | Nhỏ |
| 60 | `<fill>` | `<fill>` | Nhỏ |
| 100 | `<fill>` | `<fill>` | Trung bình |
| 150 | `<fill>` | `<fill>` | Ứng viên |
| 220 | `<fill>` | `<fill>` | Ứng viên |
| 300 | `<fill>` | `<fill>` | Ứng viên |

Sau khi tìm được N, dùng `2N` cho speedup theo yêu cầu.

### 7.4. Granularity và load balance

Chạy cùng N, cùng P, thay đổi tile grid:

```text
1x1, 2x2, 4x3, 5x4
```

Ý nghĩa:

- `1x1`: task thô, ít giao tiếp, nhưng khó chia đều.
- `2x2`: cân bằng hơn, chi phí giao tiếp tăng vừa phải.
- `4x3`: nhiều task, dynamic scheduling phát huy tốt hơn.
- `5x4`: task rất mịn, dễ cân bằng hơn nhưng có nguy cơ tăng overhead và false positive.

Kết quả lấy từ:

```text
granularity/granularity_overview.csv
granularity/grid_<grid>/rank_metrics_stacked.png
```

Nếu `idle_gap_ratio > 0.25`, hệ thống chưa cân bằng tải theo tiêu chí của thầy. Khi đó cần giảm/tăng độ mịn và chạy lại.

### 7.5. Speedup

Chạy với `2N`, thay đổi số process:

```text
P = 1, 2, 4, 8, 12
```

Speedup:

```text
S(P) = T(1) / T(P)
Efficiency(P) = S(P) / P
```

Kết quả lấy từ:

```text
speedup/raw/speedup.csv
speedup/figures/speedup.png
```

Khi P tăng, speedup có thể không tuyến tính vì:

- YOLO worker có overhead khởi tạo.
- Dữ liệu task và bbox phải truyền qua LAN.
- Một số rank có thể chờ task hoặc chờ kết quả.
- Máy trong cụm không đồng nhất.

### 7.6. Demo live camera

Demo live không phải benchmark chính nhưng giúp bài trình bày hấp dẫn. Master lấy camera, hiển thị kết quả realtime. Để giảm detect thừa trong live, nhóm dùng full-frame anchor:

```text
YOLO_LIVE_ANCHOR_POLICY=anchor-gate
```

Lưu ý khi báo cáo: live camera là demo ứng dụng; kết quả benchmark chính vẫn là offline MOT17 CPU/MPI.

---

## 8. Nhận Xét, Hạn Chế Và Hướng Phát Triển

### 8.1. Ưu điểm

- Bài toán có tính thực tế và dễ demo.
- Thuật toán song song rõ ràng: task-level, hybrid temporal-spatial decomposition.
- Có hai scheduler static/dynamic để so sánh.
- Có đo correctness, accuracy, load balance và speedup.
- Chạy trên ba máy vật lý, đúng yêu cầu không dùng cloud.

### 8.2. Hạn chế

- YOLO pretrained không đảm bảo đếm đúng tuyệt đối trên MOT17, nhất là scene đông người và occlusion.
- Chia tile quá mịn có thể làm tăng false positive nếu hậu xử lý không đủ tốt.
- CPU YOLO khó đạt 30 FPS trong live demo.
- Cụm MacBook không đồng nhất nên speedup phụ thuộc mạnh vào máy chậm nhất và LAN.

### 8.3. Hướng phát triển

- Dùng tracker như ByteTrack/DeepSORT để ổn định count theo thời gian.
- Fine-tune model cho scene người đi bộ.
- Dùng non-blocking communication nhiều hơn để giảm thời gian chờ.
- Dùng work stealing hoặc weighted dynamic scheduling cho cụm không đồng nhất.
- Tách benchmark CPU và demo GPU/MPS rõ ràng hơn.

---

## 9. Kết Luận

Nhóm đã xây dựng chương trình C++17/OpenMPI để song song hóa pipeline đếm người trong video bằng YOLO. Thuật toán sử dụng task-level parallelism với hybrid temporal-spatial decomposition. Dynamic scheduling giúp cải thiện cân bằng tải khi thời gian xử lý từng tile không đồng đều.

Về mặt yêu cầu môn học, đề tài đáp ứng các điểm chính: chạy trên cụm ba máy vật lý, có thuật toán song song rõ ràng, có baseline serial, có ground truth MOT17, có đo correctness, N, granularity/load balance và speedup. Demo live camera là phần bổ sung để minh họa ứng dụng thực tế, còn kết quả benchmark chính dùng CPU/MPI offline để đúng bản chất parallel programming.

---

## 10. Tài Liệu Tham Khảo

1. Open MPI Project, *Open MPI Documentation*, https://docs.open-mpi.org/  
2. MOTChallenge, *MOT17 Benchmark*, https://motchallenge.net/data/MOT17/  
3. Ultralytics, *YOLO Documentation*, https://docs.ultralytics.com/  
4. Joseph Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, 2016.  
5. Milan et al., *MOT16: A Benchmark for Multi-Object Tracking*, 2016.  
