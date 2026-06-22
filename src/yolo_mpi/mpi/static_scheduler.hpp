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
    std::vector<Detection> detections;
    DetectorRunner detector(cfg, rank);

    for (const auto& task : tasks) {
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

    if (out_metrics) {
        *out_metrics = m;
    }

    std::string payload = serialize_detections(detections);

    if (include_metrics) {
        payload += serialize_metrics(m);
    }

    return payload;
}

// Run static MPI mode: each rank works independently, then root gathers results.
static std::string run_static(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    int rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    auto local_tasks = select_static_tasks(tasks, rank, world_size, cfg.chunk_size);

    Metrics local_metrics;
    auto payload = process_tasks_payload(cfg, local_tasks, rank, &local_metrics, false);

    auto c0 = std::chrono::steady_clock::now();
    std::string gathered = gather_string(payload, 0, comm);
    auto c1 = std::chrono::steady_clock::now();

    // Static mode communicates mainly at the final gather step.
    local_metrics.comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();

    // Gather metrics separately so communication time is included in rank_metrics.csv.
    std::string gathered_metrics = gather_string(serialize_metrics(local_metrics), 0, comm);

    if (rank == 0) {
        return gathered + gathered_metrics;
    }

    return "";
}
