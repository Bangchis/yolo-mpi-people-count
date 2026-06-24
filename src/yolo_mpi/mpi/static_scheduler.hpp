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

// Physical-node information used by weighted-node mapping.
struct NodeLayout {
    MPI_Comm node_comm = MPI_COMM_NULL;
    int node_local_rank = 0;
    int node_local_size = 1;
    int node_rank = 0;
    int node_count = 1;
    int node_leader_world_rank = 0;
    std::vector<int> node_leaders;
    std::vector<int> node_weights;
};

// Build one communicator per physical machine and derive node weights from ranks per node.
static NodeLayout build_node_layout(MPI_Comm comm) {
    NodeLayout layout;
    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &layout.node_comm);
    MPI_Comm_rank(layout.node_comm, &layout.node_local_rank);
    MPI_Comm_size(layout.node_comm, &layout.node_local_size);

    int local_leader = world_rank;
    MPI_Allreduce(&world_rank, &local_leader, 1, MPI_INT, MPI_MIN, layout.node_comm);
    layout.node_leader_world_rank = local_leader;

    std::vector<int> leader_per_rank(world_size, 0);
    std::vector<int> size_per_rank(world_size, 1);

    MPI_Allgather(&local_leader, 1, MPI_INT, leader_per_rank.data(), 1, MPI_INT, comm);
    MPI_Allgather(&layout.node_local_size, 1, MPI_INT, size_per_rank.data(), 1, MPI_INT, comm);

    for (int rank = 0; rank < world_size; ++rank) {
        int leader = leader_per_rank[rank];

        if (std::find(layout.node_leaders.begin(), layout.node_leaders.end(), leader) != layout.node_leaders.end()) {
            continue;
        }

        layout.node_leaders.push_back(leader);
        layout.node_weights.push_back(size_per_rank[rank]);
    }

    layout.node_count = static_cast<int>(layout.node_leaders.size());

    for (int i = 0; i < layout.node_count; ++i) {
        if (layout.node_leaders[i] == local_leader) {
            layout.node_rank = i;
            break;
        }
    }

    return layout;
}

// Weighted owner for one frame. A 4/6/2 rank placement becomes a 4/6/2 frame placement.
static int owner_node_for_frame(int frame_offset, const std::vector<int>& node_weights) {
    int total_weight = std::accumulate(node_weights.begin(), node_weights.end(), 0);
    int slot = frame_offset % std::max(1, total_weight);
    int prefix = 0;

    for (int node = 0; node < static_cast<int>(node_weights.size()); ++node) {
        prefix += node_weights[node];

        if (slot < prefix) {
            return node;
        }
    }

    return static_cast<int>(node_weights.size()) - 1;
}

// Count how many earlier frames were assigned to this node.
static int owned_frames_before(int frame_offset, int node_rank, const std::vector<int>& node_weights) {
    int total_weight = std::accumulate(node_weights.begin(), node_weights.end(), 0);
    total_weight = std::max(1, total_weight);

    int full_cycles = frame_offset / total_weight;
    int remainder = frame_offset % total_weight;
    int prefix = 0;

    for (int node = 0; node < node_rank; ++node) {
        prefix += node_weights[node];
    }

    int weight = node_weights[node_rank];
    int extra = std::max(0, std::min(remainder, prefix + weight) - prefix);
    return full_cycles * weight + extra;
}

// Weighted-node mapping: assign complete frames to machines, then tiles to ranks inside that machine.
static std::vector<Task> select_weighted_node_tasks(
    const Config& cfg,
    const std::vector<Task>& tasks,
    const NodeLayout& layout
) {
    auto [cols, rows] = parse_tile_grid(cfg.tile_grid);
    int tiles_per_frame = cols * rows;
    std::vector<Task> selected;

    for (const auto& task : tasks) {
        int frame_offset = task.frame_id - cfg.start_frame;
        int owner_node = owner_node_for_frame(frame_offset, layout.node_weights);

        if (owner_node != layout.node_rank) {
            continue;
        }

        int local_frame_index = owned_frames_before(frame_offset, layout.node_rank, layout.node_weights);
        int local_task_id = local_frame_index * tiles_per_frame + task.tile_id;
        int block_id = local_task_id / std::max(1, cfg.chunk_size);

        if (block_id % layout.node_local_size == layout.node_local_rank) {
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

// Keep only frame-level boxes inside one node before sending across LAN.
static std::string reduce_detections_inside_node(const Config& cfg, const std::string& payload) {
    std::vector<Detection> detections;
    std::vector<Metrics> ignored_metrics;
    parse_payload(payload, detections, ignored_metrics);

    auto by_frame = nms_by_frame(cfg, detections);
    std::vector<Detection> reduced;

    for (const auto& [frame, frame_detections] : by_frame) {
        (void)frame;
        reduced.insert(reduced.end(), frame_detections.begin(), frame_detections.end());
    }

    return serialize_detections(reduced);
}

// Choose blocking or non-blocking gather for a communicator-local payload.
static std::string gather_by_comm_mode(
    const std::string& local_payload,
    int root,
    MPI_Comm comm,
    const std::string& comm_mode,
    int tag_base
) {
    if (comm_mode == "nonblocking" || comm_mode == "streaming") {
        return gather_string_nonblocking(local_payload, root, comm, tag_base);
    }

    return gather_string(local_payload, root, comm);
}

// Stream only detection rows inside one communicator. Metrics are gathered later.
static std::string stream_detections_to_local_root(
    const Config& cfg,
    const std::vector<Task>& local_tasks,
    DetectorRunner& detector,
    Metrics& local_metrics,
    MPI_Comm local_comm,
    int tag_base
) {
    int local_rank = 0, local_size = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    const size_t batch_size = static_cast<size_t>(std::max(1, cfg.stream_batch_tasks));
    const size_t max_pending = static_cast<size_t>(std::max(1, cfg.stream_max_pending));

    if (local_rank != 0) {
        std::deque<StreamSend> pending_sends;
        double comm_ms = 0.0;

        for (size_t begin = 0; begin < local_tasks.size(); begin += batch_size) {
            size_t end = std::min(begin + batch_size, local_tasks.size());
            std::string payload = process_task_range_payload(cfg, local_tasks, begin, end, detector, local_metrics);

            comm_ms += stream_start_send(payload, 0, local_comm, tag_base, pending_sends);

            while (pending_sends.size() >= max_pending) {
                comm_ms += stream_wait_one_send(pending_sends);
            }
        }

        comm_ms += stream_wait_all_sends(pending_sends);
        comm_ms += stream_send_done(0, local_comm, tag_base);
        local_metrics.comm_ms += comm_ms;
        return "";
    }

    std::vector<std::string> payload_by_rank(local_size);
    std::deque<StreamReceive> pending_receives;
    int active_senders = std::max(0, local_size - 1);
    double comm_ms = 0.0;

    for (size_t begin = 0; begin < local_tasks.size(); begin += batch_size) {
        size_t end = std::min(begin + batch_size, local_tasks.size());
        payload_by_rank[0] += process_task_range_payload(cfg, local_tasks, begin, end, detector, local_metrics);

        auto c0 = std::chrono::steady_clock::now();
        stream_poll_root(local_comm, tag_base, active_senders, pending_receives, payload_by_rank);
        auto c1 = std::chrono::steady_clock::now();
        comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    comm_ms += stream_drain_root(local_comm, tag_base, active_senders, pending_receives, payload_by_rank);
    local_metrics.comm_ms += comm_ms;

    std::string gathered;
    for (const auto& payload : payload_by_rank) {
        gathered += payload;
    }

    return gathered;
}

// Weighted-node hierarchical scheduler: local node aggregation, then global leader gather.
static std::string run_weighted_node_static(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    int world_rank = 0;
    MPI_Comm_rank(comm, &world_rank);

    NodeLayout layout = build_node_layout(comm);
    auto local_tasks = select_weighted_node_tasks(cfg, tasks, layout);

    Metrics local_metrics;
    local_metrics.rank = world_rank;
    local_metrics.hostname = hostname();

    DetectorRunner detector(cfg, world_rank);
    std::string node_detection_payload;

    if (cfg.comm_mode == "streaming") {
        node_detection_payload = stream_detections_to_local_root(
            cfg,
            local_tasks,
            detector,
            local_metrics,
            layout.node_comm,
            4000
        );
    } else {
        std::string local_detection_payload =
            process_task_range_payload(cfg, local_tasks, 0, local_tasks.size(), detector, local_metrics);

        auto c0 = std::chrono::steady_clock::now();
        node_detection_payload = gather_by_comm_mode(
            local_detection_payload,
            0,
            layout.node_comm,
            cfg.comm_mode,
            4000
        );
        auto c1 = std::chrono::steady_clock::now();
        local_metrics.comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    int is_node_leader = layout.node_local_rank == 0 ? 1 : 0;
    MPI_Comm leader_comm = MPI_COMM_NULL;
    MPI_Comm_split(comm, is_node_leader ? 0 : MPI_UNDEFINED, world_rank, &leader_comm);

    std::string global_detection_payload;

    if (is_node_leader) {
        std::string reduced_node_payload = reduce_detections_inside_node(cfg, node_detection_payload);

        auto c0 = std::chrono::steady_clock::now();
        global_detection_payload = gather_by_comm_mode(
            reduced_node_payload,
            0,
            leader_comm,
            cfg.comm_mode,
            4300
        );
        auto c1 = std::chrono::steady_clock::now();
        local_metrics.comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    }

    // Gather metrics after the main detection payload so leader rows include global communication time.
    std::string local_metric_payload = serialize_metrics(local_metrics);
    std::string node_metric_payload = gather_by_comm_mode(
        local_metric_payload,
        0,
        layout.node_comm,
        cfg.comm_mode,
        4600
    );

    std::string global_metric_payload;

    if (is_node_leader) {
        global_metric_payload = gather_by_comm_mode(
            node_metric_payload,
            0,
            leader_comm,
            cfg.comm_mode,
            4900
        );
    }

    if (leader_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&leader_comm);
    }

    if (layout.node_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&layout.node_comm);
    }

    if (world_rank == 0) {
        return global_detection_payload + global_metric_payload;
    }

    return "";
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
    if (cfg.mapping == "weighted-node") {
        return run_weighted_node_static(cfg, tasks, comm);
    }

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
