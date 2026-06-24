// Static scheduling: flatten all frame/tile tasks and assign block-cyclically.
// It is simple and low-communication, but can be imbalanced on hard frames.

// Select the subset of tasks assigned to this rank by block-cyclic mapping.
static std::vector<Task> select_static_tasks(const std::vector<Task>& tasks, int rank, int world_size, int chunk_size) {
    std::vector<Task> selected;

    for (const auto& task : tasks) {
        // chunk_size groups consecutive tasks before applying round-robin mapping.
        int block_id = task.task_id / std::max(1, chunk_size);

        if (block_id % world_size == rank) {
            // Static assignment: this rank owns this block for the whole run.
            selected.push_back(task);
        }
    }

    return selected;
}

// Process a slice of local tasks with an already-created detector.
static std::string process_task_range_payload(
    const Config& cfg,
    const std::vector<Task>& tasks,
    size_t begin,
    size_t end,
    DetectorRunner& detector,
    Metrics& m
) {
    std::vector<Detection> detections;

    for (size_t i = begin; i < end; ++i) {
        const auto& task = tasks[i];

        // compute_ms includes optional artificial sleep and detector execution.
        auto t0 = std::chrono::steady_clock::now();

        if (cfg.sleep_ms > 0) {
            // Small jitter is useful for testing load-balancing behavior.
            int jitter = (task.frame_id + task.tile_id) % 5;
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms + jitter));
        }

        // yolo_ms measures only the detector call, not local bookkeeping.
        auto y0 = std::chrono::steady_clock::now();
        auto task_dets = detector.detect(task);
        auto y1 = std::chrono::steady_clock::now();

        // Bboxes from all local tasks are serialized into one payload later.
        detections.insert(detections.end(), task_dets.begin(), task_dets.end());

        auto t1 = std::chrono::steady_clock::now();
        m.tasks_done += 1;

        if (task.tile_id == 0) {
            // Count full frames once, using tile 0 as the representative tile.
            m.frames_done += 1;
        }

        m.yolo_ms += std::chrono::duration<double, std::milli>(y1 - y0).count();
        m.compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return serialize_detections(detections);
}

// Process a local list of tasks and serialize all local detections and metrics.
static std::string process_tasks_payload(
    const Config& cfg,
    const std::vector<Task>& tasks,
    int rank,
    Metrics* out_metrics = nullptr,
    bool include_metrics = true
) {
    Metrics m;
    m.rank = rank;
    m.hostname = hostname();

    // Each rank creates its own detector backend once, then reuses it for all tasks.
    DetectorRunner detector(cfg, rank);
    std::string payload = process_task_range_payload(cfg, tasks, 0, tasks.size(), detector, m);

    if (out_metrics) {
        *out_metrics = m;
    }

    if (include_metrics) {
        payload += serialize_metrics(m);
    }

    return payload;
}

// Streaming mode: send completed batches while the next batches are still computing.
static std::string run_static_streaming(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    int rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    constexpr int root = 0;
    constexpr int tag_base = 3000;

    auto local_tasks = select_static_tasks(tasks, rank, world_size, cfg.chunk_size);
    const size_t batch_size = static_cast<size_t>(std::max(1, cfg.stream_batch_tasks));
    const size_t max_pending = static_cast<size_t>(std::max(1, cfg.stream_max_pending));

    Metrics local_metrics;
    local_metrics.rank = rank;
    local_metrics.hostname = hostname();

    DetectorRunner detector(cfg, rank);

    if (rank != root) {
        std::deque<StreamSend> pending_sends;
        double comm_ms = 0.0;

        for (size_t begin = 0; begin < local_tasks.size(); begin += batch_size) {
            size_t end = std::min(begin + batch_size, local_tasks.size());

            // Send this batch, then keep computing while MPI progresses in the background.
            std::string payload = process_task_range_payload(cfg, local_tasks, begin, end, detector, local_metrics);
            comm_ms += stream_start_send(payload, root, comm, tag_base, pending_sends);

            // Limit memory use and expose network back-pressure if root cannot receive fast enough.
            while (pending_sends.size() >= max_pending) {
                comm_ms += stream_wait_one_send(pending_sends);
            }
        }

        comm_ms += stream_wait_all_sends(pending_sends);
        local_metrics.comm_ms += comm_ms;

        // Metrics are sent as the final normal payload before the done marker.
        std::string metric_payload = serialize_metrics(local_metrics);
        double metrics_send_ms = 0.0;
        metrics_send_ms += stream_start_send(metric_payload, root, comm, tag_base, pending_sends);
        metrics_send_ms += stream_wait_all_sends(pending_sends);
        metrics_send_ms += stream_send_done(root, comm, tag_base);
        (void)metrics_send_ms;

        return "";
    }

    std::vector<std::string> payload_by_rank(world_size);
    std::deque<StreamReceive> pending_receives;
    int active_senders = std::max(0, world_size - 1);
    double comm_ms = 0.0;

    for (size_t begin = 0; begin < local_tasks.size(); begin += batch_size) {
        size_t end = std::min(begin + batch_size, local_tasks.size());

        // Root also computes its assigned tasks; it is not just a passive collector.
        payload_by_rank[root] += process_task_range_payload(cfg, local_tasks, begin, end, detector, local_metrics);

        auto c0 = std::chrono::steady_clock::now();
        stream_poll_root(comm, tag_base, active_senders, pending_receives, payload_by_rank);
        auto c1 = std::chrono::steady_clock::now();
        comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    // Root has finished computing; now drain any remaining worker batches.
    comm_ms += stream_drain_root(comm, tag_base, active_senders, pending_receives, payload_by_rank);
    local_metrics.comm_ms += comm_ms;
    payload_by_rank[root] += serialize_metrics(local_metrics);

    std::string gathered;
    for (const auto& payload : payload_by_rank) {
        gathered += payload;
    }

    return gathered;
}

// Run static MPI mode: each rank works independently, then root gathers results.
static std::string run_static(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    if (cfg.comm_mode == "streaming") {
        return run_static_streaming(cfg, tasks, comm);
    }

    int rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    auto local_tasks = select_static_tasks(tasks, rank, world_size, cfg.chunk_size);

    Metrics local_metrics;
    auto payload = process_tasks_payload(cfg, local_tasks, rank, &local_metrics, false);

    auto c0 = std::chrono::steady_clock::now();
    std::string gathered = cfg.comm_mode == "nonblocking"
        ? gather_string_nonblocking(payload, 0, comm, 2000)
        : gather_string(payload, 0, comm);
    auto c1 = std::chrono::steady_clock::now();

    // Static mode communicates mainly at the final result gather step.
    local_metrics.comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();

    // Gather metrics separately so communication time is included in rank_metrics.csv.
    std::string metric_payload = serialize_metrics(local_metrics);
    std::string gathered_metrics = cfg.comm_mode == "nonblocking"
        ? gather_string_nonblocking(metric_payload, 0, comm, 2100)
        : gather_string(metric_payload, 0, comm);

    if (rank == 0) {
        return gathered + gathered_metrics;
    }

    return "";
}
