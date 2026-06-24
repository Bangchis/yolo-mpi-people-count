// Write one count per frame for correctness checks and report plots.
static void write_frame_counts(
    const fs::path& path,
    const Config& cfg,
    const std::map<int, std::vector<Detection>>& by_frame
) {
    std::ofstream f(path);
    f << "frame_id,person_count\n";

    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        // Missing frames mean no detection survived postprocess for that frame.
        auto it = by_frame.find(frame);
        f << frame << "," << (it == by_frame.end() ? 0 : it->second.size()) << "\n";
    }
}

// Write final postprocessed bounding boxes for inspection/debugging.
static void write_bboxes(const fs::path& path, const std::map<int, std::vector<Detection>>& by_frame) {
    std::ofstream f(path);
    f << "frame_id,tile_id,rank,x1,y1,x2,y2,conf,cls\n";
    f << std::fixed << std::setprecision(4);

    for (const auto& [frame, detections] : by_frame) {
        // These boxes are already frame-level boxes after tile remap and global NMS.
        for (const auto& det : detections) {
            f << det.frame_id << "," << det.tile_id << "," << det.rank << ","
              << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << ","
              << det.conf << "," << det.cls << "\n";
        }
    }
}

// Write per-rank timing data and derive idle time from the slowest compute rank.
static void write_rank_metrics(const fs::path& path, std::vector<Metrics> metrics) {
    double max_compute = 0;

    // The slowest active rank defines the end of parallel compute time.
    for (const auto& m : metrics) {
        max_compute = std::max(max_compute, m.compute_ms);
    }

    // Idle time is estimated as waiting for the slowest rank to finish.
    for (auto& m : metrics) {
        m.idle_ms = std::max(0.0, max_compute - m.compute_ms);
    }

    // Keep output stable so plots always show rank 0, rank 1, rank 2, ...
    std::sort(metrics.begin(), metrics.end(), [](const Metrics& a, const Metrics& b) { return a.rank < b.rank; });

    std::ofstream f(path);
    f << "rank,hostname,tasks_done,frames_done,compute_ms,io_ms,yolo_ms,comm_ms,idle_ms\n";
    f << std::fixed << std::setprecision(4);

    for (const auto& m : metrics) {
        f << m.rank << "," << m.hostname << "," << m.tasks_done << "," << m.frames_done << ","
          << m.compute_ms << "," << m.io_ms << "," << m.yolo_ms << "," << m.comm_ms << "," << m.idle_ms << "\n";
    }
}

// Summarize load balance as max active compute time divided by average compute time.
static double load_imbalance(const std::vector<Metrics>& metrics) {
    std::vector<double> active;

    for (const auto& m : metrics) {
        // Ignore coordinator-only rows that did not process any task.
        if (m.tasks_done > 0) {
            active.push_back(m.compute_ms);
        }
    }

    if (active.empty()) {
        return 0;
    }

    double avg = std::accumulate(active.begin(), active.end(), 0.0) / active.size();

    if (avg <= 0) {
        return 0;
    }

    // Value near 1.0 means balanced; larger values mean one rank worked much longer.
    return *std::max_element(active.begin(), active.end()) / avg;
}

// Re-run the same tasks serially and compare final frame counts against MPI output.
static bool verify_counts(
    const Config& cfg,
    const std::vector<Task>& tasks,
    const std::map<int, std::vector<Detection>>& parallel_by_frame,
    fs::path report
) {
    std::vector<Detection> serial;
    DetectorRunner detector(cfg, -1);

    // Serial baseline processes the exact same task list with no MPI scheduling.
    for (const auto& task : tasks) {
        auto dets = detector.detect(task);
        serial.insert(serial.end(), dets.begin(), dets.end());
    }

    // Both serial and MPI outputs use the same postprocess path before comparison.
    auto serial_by_frame = nms_by_frame(cfg, serial);

    int max_error = 0;
    double sum_error = 0;

    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        // Correctness for the course report is count equality frame by frame.
        int p = parallel_by_frame.count(frame) ? static_cast<int>(parallel_by_frame.at(frame).size()) : 0;
        int s = serial_by_frame.count(frame) ? static_cast<int>(serial_by_frame.at(frame).size()) : 0;
        int err = std::abs(p - s);

        max_error = std::max(max_error, err);
        sum_error += err;
    }

    bool ok = max_error == 0;

    std::ofstream f(report);
    f << "CORRECTNESS_REPORT\n";
    f << "CORRECTNESS_PASS=" << (ok ? "YES" : "NO") << "\n";
    f << "max_count_error=" << max_error << "\n";
    f << "mean_count_error=" << (sum_error / std::max(1, cfg.frames)) << "\n";
    f << "frames_checked=" << cfg.frames << "\n";
    return ok;
}

// Write the one-row experiment summary consumed by benchmark/report scripts.
static void write_summary(
    const fs::path& path,
    const Config& cfg,
    int world_size,
    int num_tasks,
    double total_ms,
    const std::vector<Metrics>& metrics,
    const std::map<int, std::vector<Detection>>& by_frame,
    int correctness
) {
    double compute_max = 0, compute_sum = 0, comm_sum = 0, io_sum = 0, yolo_sum = 0;

    // "Without communication" runtime is approximated by the slowest compute rank.
    for (const auto& m : metrics) {
        compute_max = std::max(compute_max, m.compute_ms);
    }

    double idle_sum = 0;

    for (const auto& m : metrics) {
        // These totals feed the report plots: compute, communication, YOLO, idle.
        compute_sum += m.compute_ms;
        comm_sum += m.comm_ms;
        io_sum += m.io_ms;
        yolo_sum += m.yolo_ms;
        idle_sum += std::max(0.0, compute_max - m.compute_ms);
    }

    double avg_count = 0;

    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        // Average people count is a quick sanity signal for the chosen video.
        auto it = by_frame.find(frame);
        avg_count += (it == by_frame.end() ? 0 : it->second.size());
    }

    avg_count /= std::max(1, cfg.frames);

    std::ofstream f(path);
    f << "run_id,language,detector,model,device,imgsz,frames,tile_grid,num_tasks,overlap,"
      << "tile_owner_filter,dedup_ios,dedup_center,dedup_axis_overlap,dedup_gap,"
      << "dedup_near_camera,dedup_large_area_ratio,dedup_merge,mapping,schedule,"
      << "chunk_size,comm_mode,stream_batch_tasks,stream_max_pending,master_compute,world_size,"
      << "video_width,video_height,total_ms_with_comm,"
      << "total_ms_without_comm,compute_ms_max,compute_ms_avg,comm_ms_total,io_ms_total,"
      << "yolo_ms_total,idle_ms_total,load_imbalance,avg_count,correctness_pass\n";

    f << std::fixed << std::setprecision(4);

    f << cfg.run_id << ",C++17/OpenMPI," << cfg.detector << "," << cfg.model << "," << cfg.device << ","
      << cfg.imgsz << "," << cfg.frames << "," << cfg.tile_grid << "," << num_tasks << ","
      << cfg.overlap << "," << (cfg.tile_owner_filter ? 1 : 0) << "," << cfg.duplicate_ios << ","
      << cfg.duplicate_center << "," << cfg.duplicate_axis_overlap << "," << cfg.duplicate_gap << ","
      << (cfg.duplicate_near_camera ? 1 : 0) << "," << cfg.duplicate_large_area_ratio << ","
      << (cfg.duplicate_merge ? 1 : 0) << "," << cfg.mapping << "," << cfg.schedule << ","
      << cfg.chunk_size << "," << cfg.comm_mode << "," << cfg.stream_batch_tasks << ","
      << cfg.stream_max_pending << ","
      << (cfg.master_compute ? 1 : 0) << "," << world_size << ","
      << cfg.width << "," << cfg.height << "," << total_ms << "," << compute_max << ","
      << compute_max << "," << (compute_sum / std::max<size_t>(1, metrics.size())) << ","
      << comm_sum << "," << io_sum << "," << yolo_sum << "," << idle_sum << ","
      << load_imbalance(metrics) << "," << avg_count << ",";

    if (correctness < 0) {
        // Empty means this run did not request serial-vs-MPI verification.
        f << "";
    } else {
        f << correctness;
    }

    f << "\n";
}
