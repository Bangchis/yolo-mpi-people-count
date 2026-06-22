# Thuat Toan Song Song YOLO-MPI

Day la ban giai thich ngan de dua vao report hoac dung khi tap thuyet trinh.

## 1. Bai Toan

Cho video co `F` frame. Can dem so nguoi trong tung frame bang YOLO.

Neu cat moi frame thanh luoi `C x R`, so task la:

```text
T = F * C * R
```

Moi task:

```text
t = (frame_id, tile_id, x1, y1, x2, y2)
```

Task doc lap vi YOLO detect tren mot tile khong can ket qua cua tile khac.

## 2. Cap Do Song Song

Du an dung task-level parallelism.

Phan ra du lieu theo hai chieu:

```text
Temporal decomposition : video -> frame
Spatial decomposition  : frame -> tile
Hybrid decomposition   : task = frame + tile
```

## 3. Static Scheduling

Flatten tat ca task thanh mang 1D:

```text
tasks = [t0, t1, ..., t(T-1)]
```

Voi `P` MPI process va `chunk_size = k`:

```text
block_id(ti) = floor(i / k)
rank(ti) = block_id(ti) mod P
```

Pseudo code:

```text
for each MPI rank r in parallel:
    local_tasks = []
    for each task ti:
        if floor(i / chunk_size) mod P == r:
            local_tasks.append(ti)

    local_boxes = YOLO(local_tasks)
    send local_boxes to rank 0

rank 0:
    boxes = gather all local_boxes
    boxes = global_nms(boxes)
    count people per frame
```

Uu diem: it giao tiep, de giai thich.

Nhuoc diem: co the mat can bang tai neu task kho tap trung vao mot rank.

## 4. Dynamic Scheduling

Rank 0 lam master queue. Cac rank con lai la worker. Trong du an nay rank 0
co the vua phat viec vua compute neu `master_compute=1`.

Pseudo code:

```text
rank 0:
    next_task = 0
    send one task to each worker

    while tasks remain or workers active:
        if master_compute and task remains:
            run YOLO on one task locally

        receive result from any worker
        if task remains:
            send next task to that worker
        else:
            send STOP to that worker

    merge all boxes
    global_nms()
    count people per frame

worker rank r:
    while true:
        receive message from rank 0
        if STOP: break
        boxes = YOLO(task)
        send boxes and metrics to rank 0
```

Uu diem: can bang tai tot hon static, hop voi video vi tile dong nguoi cham hon
tile trong.

Nhuoc diem: nhieu communication hon static.

## 5. Communication Topology

Topology:

```text
star / master-worker
```

Communication offline:

```text
rank 0 -> worker : task metadata
worker -> rank 0 : bbox + metrics
```

Communication live:

```text
rank 0 -> worker : JPEG tile
worker -> rank 0 : bbox + metrics
```

MPI call chinh:

```text
MPI_Send
MPI_Recv
MPI_Gather
MPI_Gatherv
MPI_Iprobe
```

## 6. Gop Ket Qua Va Dem Nguoi

Moi rank tra ve bbox trong toa do frame goc. Rank 0 loc trung:

1. Tile-owner filter: giu bbox co tam nam trong vung tile goc.
2. Global NMS: sap xep bbox theo confidence, giu bbox tot nhat.
3. Cross-tile de-dup: gop bbox cua cung mot nguoi bi cat qua bien tile.
4. Count: so bbox con lai trong moi frame la so nguoi du doan.

Cong thuc IoU:

```text
IoU(A, B) = area(A intersect B) / area(A union B)
```

Neu:

```text
IoU >= threshold
```

thi hai bbox duoc xem la trung nhau.

## 7. Metric Can Do Cho Report

Correctness:

```text
serial YOLO count == MPI YOLO count
```

Accuracy voi ground truth:

```text
error(frame) = abs(pred_count(frame) - gt_count(frame))
MAE = mean(error)
RMSE = sqrt(mean(error^2))
```

Find N:

```text
Chay F = 30, 60, 100, 150, ...
Chon N sao cho runtime khoang 2-3 phut
```

Granularity:

```text
So sanh tile_grid = 1x1, 2x2, 4x3
Ve compute_ms + comm_ms + idle_ms theo rank
```

Speedup:

```text
S(P) = T(1) / T(P)
P = 1, 2, 4, 8, 12
```

Benchmark chinh nen dung CPU de dung yeu cau OpenMPI/process/core. MPS live
camera dung lam demo phu vi nhieu MPI process tren cung mot Mac co the tranh
cung Apple GPU.
