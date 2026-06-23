# Defense Q&A Notes

Use this file to prepare answers for the presentation. Keep answers short and technical.

## 1. Bài toán của nhóm là gì?

Nhóm song song hóa bài toán đếm người trong video bằng YOLO. Input là video, output là số người theo từng frame và bounding box sau hậu xử lý.

Trọng tâm không phải huấn luyện YOLO, mà là song song hóa pipeline inference video trên cụm 3 máy bằng C++17/OpenMPI.

## 2. Song song cấp độ nào?

Song song cấp tác vụ.

Một task là:

```text
(frame_id, tile_id, x1, y1, x2, y2)
```

Mỗi task chạy YOLO trên một vùng ảnh của một frame.

## 3. Phân rã loại gì?

Hybrid decomposition:

- Temporal decomposition: video -> frame.
- Spatial decomposition: frame -> tile.

Tổng số task:

```text
T = F * C * R
```

Với `F` là số frame, `C x R` là lưới tile.

## 4. Mapping technique là gì?

Lưới tile 2D được flatten thành danh sách task 1D:

```text
task_id = frame_id * (cols * rows) + tile_id
```

Static scheduling dùng block-cyclic:

```text
rank = floor(task_id / chunk_size) mod P
```

Dynamic scheduling dùng master queue: rank nào rảnh nhận task tiếp theo.

## 5. Communication topology là gì?

Master-worker/star topology.

```text
rank 0: master/coordinator
rank 1..P-1: worker
```

Dynamic communication:

```text
rank 0 -> worker: task metadata
worker -> rank 0: bbox + metrics
rank 0 -> worker: next task or STOP
```

MPI calls:

```text
MPI_Send, MPI_Recv, MPI_Gather, MPI_Gatherv, MPI_Iprobe
```

## 6. Blocking hay non-blocking?

Phần gửi/nhận chính là blocking send/receive. Rank 0 dùng thêm `MPI_Iprobe` để kiểm tra worker đã có kết quả chưa, nhờ đó có thể vừa điều phối vừa tự compute khi `master_compute=1`.

## 7. Có cân bằng tải không?

Có áp dụng load balancing bằng:

- Dynamic scheduling.
- Thử nhiều granularity: `1x1`, `2x2`, `4x3`, `5x4`.
- Weighted mapping trên cụm không đồng nhất: `8/10/6` rank thay vì `8/8/8`.

Kết quả cho thấy dataset nhỏ và vừa vẫn chưa đạt tiêu chí idle gap dưới 25%, vì overhead communication lớn, task YOLO không đều giữa các tile/frame, và 3 máy không đồng nhất. Đây là điểm cần nói thẳng: nhóm có đo load balance, kết quả chưa lý tưởng, và đã phân tích nguyên nhân.

## 8. Vì sao dynamic scheduling không nhanh hơn static trong bảng?

Vì input mini tương đối nhỏ, task nhiều nhưng mỗi task không đủ nặng so với overhead điều phối. Dynamic phải gửi task và nhận result payload nhiều lần qua rank 0, nên có thể chậm hơn static.

Kết luận đúng là: dynamic không luôn nhanh hơn, nhưng tổng quát hơn cho video irregular và cụm máy không đồng nhất.

Rerun bổ sung với 600 frame trên đủ 3 máy cũng cho kết quả tương tự:

```text
static  -> 40.619s
dynamic -> 102.316s
```

Vì vậy khi bảo vệ không nên nói dynamic nhanh hơn; nên nói dynamic là chiến lược tổng quát hơn để xử lý task không đều, nhưng overhead điều phối có thể làm runtime xấu hơn.

## 9. Correctness được kiểm tra thế nào?

So sánh serial YOLO và MPI YOLO với cùng video, model, threshold, tile grid và hậu xử lý.

Kết quả:

```text
correctness_pass = YES
mismatched_frames = 0
max_abs_error = 0
```

Điều này chứng minh phần song song hóa không làm thay đổi kết quả so với bản tuần tự.

## 10. YOLO đếm sai ground truth thì có phải lỗi MPI không?

Không.

Có hai loại đánh giá:

- Serial vs MPI: kiểm tra correctness của song song hóa.
- YOLO vs MOT17 ground truth: kiểm tra accuracy của mô hình nhận diện.

YOLO pretrained bị undercount trên MOT17 vì đám đông, occlusion và người nhỏ. Đây là sai số model, không phải lỗi MPI.

## 11. N được chọn thế nào?

Chọn `P=12`, CPU/OpenMPI, chạy nhiều kích thước input.

Kết quả với `tile_grid=5x4`:

```text
300 frame -> 54.764s
600 frame -> 123.667s
837 frame -> 146.316s
```

Chọn:

```text
N = 600 frame
```

vì runtime nằm trong khoảng 2-3 phút.

## 12. Speedup được đo thế nào?

Sau khi chọn `N=600`, nhóm tạo input `2N=1200 frame`, rồi chạy:

```text
P = 1, 2, 4, 8, 12
```

Kết quả chính:

```text
P=12 speedup with communication    = 1.939x
P=12 speedup without communication = 2.011x
```

Speedup không tuyến tính do communication overhead, rank 0 gom kết quả, YOLO worker startup, heterogeneity và CPU contention.

Rerun bổ sung cho `2N=1200` có hai sweep đầy đủ. Runtime `P=12` rất ổn định:

```text
original report sweep -> 178.306s
new repeat            -> 178.043s
mean                  -> 178.174s
```

Speedup dao động hơn runtime vì baseline `P=1` dao động mạnh giữa các lần chạy. Nếu thầy hỏi, nói rằng throughput cluster ở `P=12` ổn định, còn speedup phụ thuộc vào serial baseline trên laptop đang dùng thực tế.

## 12b. Nếu thầy hỏi chạy một máy local-only thì sao?

Có chạy thêm local-only trên master với 600 frame và grid `5x4`.

Kết quả tốt nhất trong rerun là `P=2`, khoảng `288.494s`. Khi tăng lên `P=4` và `P=8`, runtime không tốt hơn do tranh CPU, worker overhead và scheduling overhead trên cùng một máy.

Rerun bổ sung trên đủ 3 máy với 600 frame cho kết quả `P=12` khoảng `90.900s`, nhanh hơn đáng kể so với local-only tốt nhất `P=2` khoảng `288.494s`. Cách nói gọn: một máy local-only dùng nhiều process không đảm bảo scale tốt; phân tán sang nhiều máy vật lý giúp giảm áp lực trên một CPU, dù vẫn có network overhead.

## 12c. Vì sao P=2 và P=4 trong cluster sweep lại không nhanh?

Vì hostfile core có dạng:

```text
master slots=4
node1 slots=4
node2 slots=4
```

OpenMPI fill slot từ master trước. Do đó `P=1`, `P=2`, `P=4` chủ yếu vẫn chạy trên master. Khi lên `P=8`, node1 mới tham gia; khi lên `P=12`, đủ cả master, node1, node2. Đây là lý do kết quả speedup mới có dạng:

```text
P=1  -> 202.715s
P=2  -> 204.701s
P=4  -> 244.848s
P=8  -> 133.360s
P=12 -> 90.900s
```

Điểm cần nhấn mạnh: speedup tốt hơn khi workload thật sự được phân tán sang nhiều máy vật lý, không chỉ tăng số process trên một máy.

## 13. Vì sao không dùng GPU/MPS cho benchmark chính?

Vì yêu cầu môn học tập trung vào MPI process và CPU/core. Nếu dùng nhiều MPI process trên cùng một Mac với MPS, các process sẽ tranh cùng Apple GPU, làm benchmark khó giải thích theo số core.

Vì vậy:

- Benchmark chính: CPU/OpenMPI.
- Live camera/MPS: demo ứng dụng phụ.

## 14. Demo live camera có vai trò gì?

Demo live camera cho thấy pipeline ứng dụng có thể chạy realtime-ish: master lấy camera, chia frame, gửi task cho các máy, gom bbox và hiển thị kết quả.

Nhưng demo live không thay thế benchmark chính trong báo cáo.

## 15. Nếu thầy hỏi điểm khó của đề tài?

Điểm khó nằm ở:

- Chạy thật trên cụm 3 máy vật lý.
- Kết hợp C++17/OpenMPI với YOLO worker.
- Chia video theo cả thời gian và không gian.
- Hậu xử lý bbox trùng giữa các tile.
- Đo đầy đủ correctness, N, granularity, speedup, load balance.
- Phân tích cụm không đồng nhất bằng weighted mapping.
