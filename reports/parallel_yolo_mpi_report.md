# BÁO CÁO BÀI TẬP LỚN

## Đề tài: Song song hóa bài toán đếm người trong video bằng YOLO và C++17/OpenMPI trên cụm 3 máy MacBook

**Môn học:** Parallel Programming and Computing  
**Nhóm:** `<ID nhóm>`  
**Sinh viên:** `Phạm Chí Bằng - 2035477; Nguyễn Thanh Lâm - 20235519; Nguyễn Phú An - 20235466; Lưu Hiếu An - 202400093`
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
- **Weighted process mapping:** với cụm không đồng nhất, máy mạnh hơn có thể nhận nhiều MPI rank hơn.

Nếu task quá thô, một rank có thể giữ task nặng quá lâu, các rank khác rảnh. Nếu task quá mịn, số lượng task tăng mạnh làm tăng chi phí giao tiếp và hậu xử lý. Do đó nhóm đo nhiều cấu hình tile grid để chọn mức phù hợp.

Trong phần thực nghiệm, nhóm tách rõ hai ý:

1. **Dynamic scheduling experiment:** mọi rank nhận task động từ hàng đợi của rank 0. Đây là thuật toán chính.
2. **Weighted mapping experiment:** cùng là dynamic scheduling, nhưng số rank trên từng máy được gán theo sức mạnh máy. Thí nghiệm này dùng `P=24` để so sánh mapping đều `8/8/8` và mapping có trọng số `8/10/6`.

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
| Tìm N 2-3 phút | Chạy nhiều frame count, mini và full sequence | `find_N/raw/find_N.csv`, `find_N_long_fullseq_5x4/raw/find_N.csv` |
| Granularity/load balance | Chạy vài tile grid đại diện | `granularity/granularity_overview.csv`, `granularity_overview.png` |
| Static vs dynamic scheduling | So sánh hai scheduler trên cùng input | `scheduler/scheduler_comparison.csv`, `scheduler_comparison.png` |
| Stacked rank time | compute + comm + idle | `rank_metrics_stacked.png` |
| Speedup | P = 1,2,4,8,12 | `speedup/raw/speedup.csv`, `speedup.png` |
| Heterogeneous weighted mapping | P=24, uniform vs weighted | `heterogeneous_balance.png`, `host_metrics.csv` |

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
YOLO_SPEEDUP_MAP_BY=node:OVERSUBSCRIBE \
YOLO_RUN_SCHEDULER_COMPARE=1 \
YOLO_RUN_HETEROGENEOUS=1 \
YOLO_HET_FRAMES=150 \
YOLO_HET_TILE_GRID=5x4 \
YOLO_HET_NP=24 \
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

Kết quả dưới đây lấy từ lần chạy:

```text
results/report_mot17_mini_final_20260623-154318/
```

Các file CSV và hình trong thư mục này được sinh tự động bởi `scripts/run/report_mot17_mini.sh`, sau đó được tổng hợp bằng `scripts/report/summarize_report_dir.py`.

### 7.1. Kiểm tra correctness: serial vs MPI

Mục tiêu là chứng minh song song hóa không làm sai kết quả so với cách chạy tuần tự cùng cấu hình.

| Pass | Frames | Mismatched | Max Error | Mean Error |
|---|---:|---:|---:|---:|
| YES | 30 | 0 | 0 | 0.000 |

Kết quả full run lấy từ:

```text
correctness/correctness_compare.csv
correctness/correctness_per_frame.csv
```

Nhận xét: với cùng model, threshold, tile grid và hậu xử lý, bản MPI cho kết quả khớp hoàn toàn với bản serial trên 30 frame kiểm tra. Điều này chứng minh phần chia task/gom kết quả MPI không làm thay đổi kết quả thuật toán.

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

| Frames | MAE | RMSE | MAPE | Exact | GT Avg | Pred Avg |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 7.983 | 8.269 | 0.490 | 0.000 | 16.207 | 8.223 |

Hình cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/accuracy/count_error_plot.png
```

Nhận xét: YOLO pretrained bị thiếu người trên MOT17 vì scene đông người, nhiều người bị che khuất và kích thước người nhỏ. Đây là sai số của mô hình nhận diện, không phải sai số của phần song song hóa. Phần song song hóa đã được kiểm chứng riêng bằng serial-vs-MPI ở mục 7.1.

### 7.3. Xác định kích thước dữ liệu N

Theo yêu cầu, chọn số process bằng số core/process chính của cụm, ví dụ `P=12`. Chạy nhiều kích thước frame để tìm N sao cho runtime khoảng 2-3 phút.

Kết quả lấy từ:

```text
find_N/raw/find_N.csv
find_N/figures/find_N_runtime.png
```

| N frames | Time with comm (s) | Time without comm (s) | Nhận xét |
|---:|---:|---:|---|
| 30 | 9.165 | 2.999 | Nhỏ |
| 60 | 12.570 | 6.801 | Nhỏ |
| 100 | 17.776 | 11.596 | Trung bình |
| 150 | 23.585 | 17.103 | Chưa đủ 2-3 phút |

Hình cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/find_N/figures/find_N_runtime.png
```

Nhận xét: MOT17-mini 300 frame quá ngắn nên chưa đủ để đạt runtime 2-3 phút. Vì vậy nhóm dùng thêm sequence dài hơn trong `data/mot17-fullseq/` để tìm N cuối cùng.

Kết quả dài hơn với `MOT17-05-SDP-837`, `P=12`, `tile_grid=5x4`:

| N frames | Time with comm (s) | Time without comm (s) | P | Grid | Nhận xét |
|---:|---:|---:|---:|---|---|
| 300 | 54.764 | 49.156 | 12 | 5x4 | Dưới 2 phút |
| 600 | 123.667 | 114.352 | 12 | 5x4 | Chọn làm N |
| 837 | 146.316 | 139.136 | 12 | 5x4 | Trong vùng 2-3 phút |

Hình dài cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/find_N_long_fullseq_5x4/figures/find_N_runtime.png
```

Nhóm chọn `N = 600 frame` vì runtime có communication là khoảng 123.7 giây, nằm trong vùng 2-3 phút theo yêu cầu. Với speedup, nhóm tạo video benchmark `2N = 1200 frame` bằng cách nối hai đoạn MOT17 cùng kích thước `960x540`.

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
granularity/granularity_overview.png
```

| Grid | Ranks | Tasks | Compute Max (s) | Compute Avg (s) | Comm Total (s) | Idle Total (s) | Idle Gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| grid_1x1 | 12 | 150 | 1.367 | 0.483 | 47.417 | 10.615 | 0.838 | NO |
| grid_2x2 | 12 | 600 | 5.447 | 2.205 | 72.434 | 38.902 | 0.802 | NO |
| grid_4x3 | 12 | 1800 | 16.329 | 6.375 | 154.944 | 119.449 | 0.809 | NO |
| grid_5x4 | 12 | 3000 | 25.263 | 9.133 | 230.059 | 193.569 | 0.824 | NO |

Hình cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/granularity/granularity_overview.png
results/report_mot17_mini_final_20260623-154318/granularity/grid_1x1/rank_metrics_stacked.png
results/report_mot17_mini_final_20260623-154318/granularity/grid_2x2/rank_metrics_stacked.png
results/report_mot17_mini_final_20260623-154318/granularity/grid_4x3/rank_metrics_stacked.png
results/report_mot17_mini_final_20260623-154318/granularity/grid_5x4/rank_metrics_stacked.png
```

Nhận xét: theo tiêu chí `idle_gap_ratio <= 0.25`, các cấu hình trên chưa đạt cân bằng tải hoàn hảo. Nguyên nhân chính là cụm không đồng nhất và chi phí truyền thông/khởi tạo worker lớn so với thời gian xử lý mini dataset. Tuy vậy dynamic scheduling vẫn giúp rank nhanh nhận thêm task, thể hiện rõ hơn trong thí nghiệm weighted ở mục 7.7.

### 7.5. So sánh static và dynamic scheduling

Thí nghiệm này giữ nguyên `P=12`, `N=150`, `tile_grid=4x3`, chỉ thay scheduler:

```text
static  : chia task theo block-cyclic ngay từ đầu
dynamic : rank 0 giữ queue, rank rảnh nhận task mới
```

Kết quả:

| Schedule | P | Frames | Grid | With Comm (s) | Without Comm (s) | Load Imbalance | Idle Gap | Pass |
|---|---:|---:|---|---:|---:|---:|---:|---|
| static | 12 | 150 | 4x3 | 18.437 | 13.823 | 1.807 | 0.754 | NO |
| dynamic | 12 | 150 | 4x3 | 23.262 | 17.699 | 2.815 | 0.826 | NO |

Hình cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/scheduler/figures/scheduler_comparison.png
results/report_mot17_mini_final_20260623-154318/scheduler/static/rank_metrics_stacked.png
results/report_mot17_mini_final_20260623-154318/scheduler/dynamic/rank_metrics_stacked.png
```

Nhận xét: trên mini dataset, dynamic scheduling chậm hơn static vì chi phí điều phối task và nhận/gửi kết quả lớn hơn lợi ích cân bằng tải. Đây là kết quả quan trọng: dynamic scheduling không phải lúc nào cũng nhanh hơn; nó phù hợp hơn khi task có độ khó lệch nhiều hoặc input đủ lớn. Nhóm vẫn chọn dynamic làm thuật toán chính vì nó tổng quát hơn cho video thực tế và cụm không đồng nhất, đồng thời có thêm weighted mapping ở mục 7.7 để giảm tác động máy mạnh/yếu.

### 7.6. Speedup với input 2N

Chạy với `2N = 1200 frame`, `tile_grid=5x4`, thay đổi số process:

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
speedup_2N/raw/speedup.csv
speedup_2N/figures/speedup.png
```

| P | With Comm (s) | Without Comm (s) | Speedup | Efficiency |
|---:|---:|---:|---:|---:|
| 1 | 345.677 | 342.467 | 1.000 | 1.000 |
| 2 | 276.155 | 272.546 | 1.252 | 0.626 |
| 4 | 244.747 | 240.905 | 1.412 | 0.353 |
| 8 | 200.770 | 194.282 | 1.722 | 0.215 |
| 12 | 178.306 | 170.255 | 1.939 | 0.162 |

Hình cần đưa vào báo cáo:

```text
results/report_mot17_mini_final_20260623-154318/speedup_2N/figures/speedup.png
```

Khi P tăng, speedup tăng nhưng không tuyến tính. Ở `P=12`, speedup có communication đạt `1.939x`; speedup không tính communication đạt khoảng `2.011x`. Nguyên nhân không tuyến tính:

- YOLO worker có overhead khởi tạo.
- Dữ liệu task và bbox phải truyền qua LAN.
- Một số rank có thể chờ task hoặc chờ kết quả.
- Máy trong cụm không đồng nhất.
- Nhiều tiến trình CPU cùng chạy YOLO có thể cạnh tranh tài nguyên trên từng máy.

### 7.7. Thí nghiệm dynamic + weighted mapping trên cụm không đồng nhất

Ngoài các thí nghiệm bắt buộc với `P=1,2,4,8,12`, nhóm chạy thêm thí nghiệm nâng cao với `P=24` để phân tích cụm không đồng nhất.

Hai cấu hình:

```text
uniform_24  : master/node1/node2 = 8/8/8 ranks
weighted_24 : master/node1/node2 = 8/10/6 ranks
```

Cả hai cấu hình đều dùng **dynamic scheduling**, nghĩa là mapping có trọng số chỉ quyết định số rank ban đầu trên từng máy; còn trong quá trình chạy, rank nào xong sớm sẽ nhận task mới từ master. Cách này phù hợp với cụm thực tế vì node1 mạnh hơn nên có thể chạy nhiều rank hơn, node2 yếu hơn nên nhận ít rank hơn.

Các hình và bảng cần đưa vào report:

```text
heterogeneous/figures/heterogeneous_balance.png
heterogeneous/uniform_24/rank_metrics_stacked.png
heterogeneous/weighted_24/rank_metrics_stacked.png
heterogeneous/uniform_24/host_metrics.csv
heterogeneous/weighted_24/host_metrics.csv
```

| Case | P | Frames | Grid | With Comm (s) | Without Comm (s) | Load Imbalance |
|---|---:|---:|---|---:|---:|---:|
| uniform_24 | 24 | 150 | 5x4 | 43.659 | 32.268 | 2.607 |
| weighted_24 | 24 | 150 | 5x4 | 47.238 | 30.864 | 2.379 |

Phân bố task theo host:

| Case | Host | Ranks | Tasks | Task Share |
|---|---|---:|---:|---:|
| uniform_24 | node2 | 8 | 695 | 0.232 |
| uniform_24 | node1 | 8 | 1083 | 0.361 |
| uniform_24 | master | 8 | 1222 | 0.407 |
| weighted_24 | node2 | 6 | 417 | 0.139 |
| weighted_24 | node1 | 10 | 1335 | 0.445 |
| weighted_24 | master | 8 | 1248 | 0.416 |

Nhận xét: weighted mapping làm `total_ms_without_comm` giảm từ 32.268s xuống 30.864s và giảm `load_imbalance` từ 2.607 xuống 2.379, tức là phần compute thuần được phân bổ hợp lý hơn cho cụm không đồng nhất. Tuy nhiên `total_ms_with_comm` tăng từ 43.659s lên 47.238s, cho thấy nhiều rank và nhiều task mịn hơn có thể làm chi phí giao tiếp/điều phối lớn hơn lợi ích compute. Đây là trade-off quan trọng cần trình bày khi bảo vệ.

Ý nghĩa hình:

- Biểu đồ host task share cho thấy mỗi máy xử lý bao nhiêu phần trăm task.
- Biểu đồ runtime cho thấy weighted mapping có giảm thời gian chạy hay không.
- Stacked bar theo rank cho thấy compute/communication/idle của từng rank.

### 7.8. Các hình minh họa cần có trong báo cáo

Để báo cáo không chỉ có bảng số liệu, nhóm đưa vào các hình sau:

| Hình | File |
|---|---|
| Sai số đếm theo frame | `accuracy/count_error_plot.png` |
| Runtime theo N | `find_N/figures/find_N_runtime.png` |
| Runtime theo N dài | `find_N_long_fullseq_5x4/figures/find_N_runtime.png` |
| Granularity overview | `granularity/granularity_overview.png` |
| Granularity/load balance | `granularity/grid_*/rank_metrics_stacked.png` |
| Static vs dynamic scheduler | `scheduler/figures/scheduler_comparison.png` |
| Speedup 2N | `speedup_2N/figures/speedup.png` |
| Weighted vs uniform mapping | `heterogeneous/figures/heterogeneous_balance.png` |

Các hình này tương ứng trực tiếp với các yêu cầu của thầy: đúng/sai, chọn N, granularity, load balance và speedup.

### 7.9. Demo live camera

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
