static void live_worker_loop(const Config& cfg, int rank) {
    const int task_tag = 50;
    const int stop_tag = 60;
    const int result_tag = 70;
    DetectorRunner detector(cfg, rank);
    while (true) {
        MPI_Status status;
        auto payload = recv_string(0, MPI_ANY_TAG, &status);
        if (status.MPI_TAG == stop_tag) break;
        if (status.MPI_TAG != task_tag) {
            throw std::runtime_error("worker received unexpected live MPI tag");
        }
        auto image_task = parse_image_task_payload(payload);
        auto result = process_one_image_task_payload(cfg, detector, image_task, rank);
        send_string(result, 0, result_tag);
    }
}

static void stop_live_workers(int world_size) {
    const int stop_tag = 60;
    for (int worker = 1; worker < world_size; ++worker) {
        send_string("", worker, stop_tag);
    }
}

static void run_live(const Config& cfg, int rank, int world_size) {
    if (rank != 0) {
        // Non-master ranks do not touch the camera or GUI; they only receive tiles.
        live_worker_loop(cfg, rank);
        return;
    }

    fs::create_directories(cfg.output);
    OutputPipeProcess camera(camera_source_args(cfg));
    std::unique_ptr<InputPipeProcess> viewer;
    if (cfg.live_view) {
        viewer = std::make_unique<InputPipeProcess>(viewer_args(cfg));
    }

    std::vector<Detection> all_detections;
    std::vector<Detection> previous_live_detections;
    std::vector<Metrics> all_metrics;
    std::vector<LiveFrameEvent> events;
    if (world_size > 1) {
        all_metrics.push_back(Metrics{0, hostname(), 0, 0, 0, 0, 0, 0, 0});
    }

    std::unique_ptr<DetectorRunner> local_detector;
    if (world_size == 1 || cfg.live_master_compute || cfg.live_anchor_full_frame) {
        local_detector = std::make_unique<DetectorRunner>(cfg, 0);
    }

    int total_tasks = 0;
    auto run_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cfg.frames; ++i) {
        CameraFrame frame;
        if (!read_next_camera_frame(camera, frame)) {
            break;
        }
        auto frame_t0 = std::chrono::steady_clock::now();
        std::vector<Detection> frame_detections;
        if (world_size == 1) {
            frame_detections = process_live_frame_locally(cfg, *local_detector, frame, all_metrics);
        } else {
            frame_detections = process_live_frame_distributed(
                cfg,
                (cfg.live_master_compute && !cfg.live_anchor_full_frame) ? local_detector.get() : nullptr,
                frame,
                all_metrics,
                world_size
            );
        }
        if (cfg.live_anchor_full_frame) {
            auto anchor_detections = process_live_frame_anchor(cfg, *local_detector, frame, all_metrics);
            frame_detections = merge_anchor_and_tile_detections(
                cfg,
                anchor_detections,
                frame_detections,
                frame.width,
                frame.height
            );
        }
        frame_detections = temporal_dedup_against_previous(
            cfg,
            frame_detections,
            previous_live_detections,
            frame.width,
            frame.height
        );
        previous_live_detections = frame_detections;
        auto frame_t1 = std::chrono::steady_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(frame_t1 - frame_t0).count();

        all_detections.insert(all_detections.end(), frame_detections.begin(), frame_detections.end());
        total_tasks += static_cast<int>(frame.tiles.size());
        events.push_back(LiveFrameEvent{
            frame.frame_id,
            static_cast<int>(frame_detections.size()),
            static_cast<int>(frame.tiles.size()),
            frame.capture_ms,
            latency_ms,
        });

        std::cout << "LIVE_FRAME frame=" << frame.frame_id
                  << " people=" << frame_detections.size()
                  << " tasks=" << frame.tiles.size()
                  << " latency_ms=" << std::fixed << std::setprecision(2) << latency_ms
                  << "\n";
        std::cout.flush();

        if (viewer) {
            send_frame_to_viewer(*viewer, frame, frame_detections);
        }
    }
    auto run_t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(run_t1 - run_t0).count();

    if (world_size > 1) {
        stop_live_workers(world_size);
    }

    all_metrics = aggregate_metrics_by_rank(all_metrics);
    auto by_frame = nms_by_frame(cfg, all_detections);
    write_frame_counts(fs::path(cfg.output) / "frame_counts.csv", cfg, by_frame);
    write_bboxes(fs::path(cfg.output) / "bboxes.csv", by_frame);
    write_rank_metrics(fs::path(cfg.output) / "rank_metrics.csv", all_metrics);
    write_live_events(fs::path(cfg.output) / "live_events.csv", events);
    write_summary(fs::path(cfg.output) / "summary.csv", cfg, world_size, total_tasks, total_ms, all_metrics, by_frame, -1);
    std::cout << "YOLO_MPI_CPP_LIVE_DONE=YES\n";
    std::cout << "RUN_DIR=" << cfg.output << "\n";
    std::cout << "SUMMARY_CSV=" << (fs::path(cfg.output) / "summary.csv").string() << "\n";
}
