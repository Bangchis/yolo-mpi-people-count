// Live camera pipeline.
// Rank 0 captures frames, distributes JPEG tiles, gathers boxes, and displays.

static std::vector<std::string> camera_source_args(const Config& cfg) {
    std::vector<std::string> args = {
        cfg.python_bin,
        cfg.camera_script,
        "--frames", std::to_string(cfg.frames),
        "--start-frame", std::to_string(cfg.start_frame),
        "--width", std::to_string(cfg.width),
        "--height", std::to_string(cfg.height),
        "--tile-grid", cfg.tile_grid,
        "--overlap", std::to_string(cfg.overlap),
        "--jpeg-quality", std::to_string(cfg.jpeg_quality),
        "--target-fps", std::to_string(cfg.live_fps),
    };
    if (!cfg.live_video_source.empty()) {
        args.push_back("--video-source");
        args.push_back(cfg.live_video_source);
    } else {
        args.push_back("--camera-index");
        args.push_back(std::to_string(cfg.camera_index));
    }
    return args;
}

static std::vector<std::string> viewer_args(const Config& cfg) {
    return {
        cfg.python_bin,
        cfg.viewer_script,
        "--display", cfg.live_view ? "1" : "0",
        "--save-dir", (fs::path(cfg.output) / "live_frames").string(),
    };
}

static bool parse_camera_frame_line(const std::string& line, CameraFrame& frame) {
    std::istringstream iss(line);
    std::string tag;
    if (!(iss >> tag >> frame.frame_id >> frame.width >> frame.height >> frame.capture_ms >> frame.encoded_frame)) {
        return false;
    }
    return tag == "FRAME";
}

static bool parse_camera_tile_line(const std::string& line, ImageTask& image_task) {
    std::istringstream iss(line);
    std::string tag;
    if (!(iss >> tag
              >> image_task.task.task_id
              >> image_task.task.frame_id
              >> image_task.task.tile_id
              >> image_task.task.x1
              >> image_task.task.y1
              >> image_task.task.x2
              >> image_task.task.y2
              >> image_task.encoded_jpeg)) {
        return false;
    }
    return tag == "TILE";
}

static bool read_next_camera_frame(OutputPipeProcess& camera, CameraFrame& frame) {
    frame = CameraFrame{};
    std::string line;
    while (camera.read_line(line)) {
        if (starts_with(line, "READY ")) {
            std::cerr << "camera_source " << line << "\n";
            continue;
        }
        if (starts_with(line, "ERROR ")) {
            throw std::runtime_error("camera source error: " + line);
        }
        if (starts_with(line, "END_STREAM ")) {
            return false;
        }
        if (starts_with(line, "FRAME ")) {
            if (!parse_camera_frame_line(line, frame)) {
                throw std::runtime_error("malformed FRAME line from camera source");
            }
            break;
        }
    }
    if (frame.encoded_frame.empty()) {
        return false;
    }

    while (camera.read_line(line)) {
        if (starts_with(line, "ERROR ")) {
            throw std::runtime_error("camera source error: " + line);
        }
        if (starts_with(line, "TILE ")) {
            ImageTask image_task;
            if (!parse_camera_tile_line(line, image_task)) {
                throw std::runtime_error("malformed TILE line from camera source");
            }
            frame.tiles.push_back(std::move(image_task));
            continue;
        }
        if (starts_with(line, "END_FRAME ")) {
            return true;
        }
    }
    return !frame.tiles.empty();
}

static std::string serialize_image_task(const ImageTask& image_task) {
    std::ostringstream out;
    const auto& task = image_task.task;
    out << "IMAGE_TASK "
        << task.task_id << " "
        << task.frame_id << " "
        << task.tile_id << " "
        << task.x1 << " "
        << task.y1 << " "
        << task.x2 << " "
        << task.y2 << " "
        << image_task.encoded_jpeg;
    return out.str();
}

static ImageTask parse_image_task_payload(const std::string& payload) {
    ImageTask image_task;
    std::istringstream iss(payload);
    std::string tag;
    if (!(iss >> tag
              >> image_task.task.task_id
              >> image_task.task.frame_id
              >> image_task.task.tile_id
              >> image_task.task.x1
              >> image_task.task.y1
              >> image_task.task.x2
              >> image_task.task.y2
              >> image_task.encoded_jpeg)) {
        throw std::runtime_error("malformed IMAGE_TASK payload");
    }
    if (tag != "IMAGE_TASK") {
        throw std::runtime_error("unexpected live task tag: " + tag);
    }
    return image_task;
}

static void send_frame_to_viewer(InputPipeProcess& viewer, const CameraFrame& frame, const std::vector<Detection>& detections) {
    std::ostringstream header;
    header << "FRAME " << frame.frame_id << " " << frame.width << " " << frame.height << " "
           << detections.size() << " " << frame.encoded_frame;
    viewer.write_line(header.str());
    for (const auto& det : detections) {
        std::ostringstream box;
        box << std::fixed << std::setprecision(4)
            << "BOX " << det.x1 << " " << det.y1 << " " << det.x2 << " " << det.y2 << " " << det.conf;
        viewer.write_line(box.str());
    }
    viewer.write_line("END_FRAME " + std::to_string(frame.frame_id));
}

static void write_live_events(const fs::path& path, const std::vector<LiveFrameEvent>& events) {
    std::ofstream f(path);
    f << "frame_id,person_count,tasks,capture_ms,latency_ms\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& event : events) {
        f << event.frame_id << "," << event.person_count << "," << event.tasks << ","
          << event.capture_ms << "," << event.latency_ms << "\n";
    }
}

static std::vector<Detection> process_live_frame_locally(
    const Config& cfg,
    DetectorRunner& detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics
) {
    std::vector<Detection> frame_detections;
    for (const auto& image_task : frame.tiles) {
        auto payload = process_one_image_task_payload(cfg, detector, image_task, 0);
        parse_payload(payload, frame_detections, metrics);
    }
    return merge_frame_detections(cfg, frame_detections, frame.width, frame.height);
}

static std::vector<Detection> process_live_frame_distributed(
    const Config& cfg,
    DetectorRunner* local_detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics,
    int world_size
) {
    const int task_tag = 50;
    const int result_tag = 70;
    const int total = static_cast<int>(frame.tiles.size());
    int next = 0;
    int completed = 0;
    int active = 0;
    std::vector<Detection> frame_detections;
    const bool master_compute = local_detector != nullptr;

    int initial_worker_tasks = std::min(world_size - 1, total);
    if (master_compute) {
        initial_worker_tasks = std::min(world_size - 1, std::max(0, total - 1));
    }
    for (int worker = 1; worker < world_size && next < initial_worker_tasks; ++worker) {
        send_string(serialize_image_task(frame.tiles[next++]), worker, task_tag);
        active += 1;
    }

    while (completed < total) {
        if (master_compute && next < total) {
            auto payload = process_one_image_task_payload(cfg, *local_detector, frame.tiles[next++], 0);
            parse_payload(payload, frame_detections, metrics);
            completed += 1;
        }

        if (active > 0) {
            MPI_Status status;
            auto payload = recv_string(MPI_ANY_SOURCE, result_tag, &status);
            parse_payload(payload, frame_detections, metrics);
            completed += 1;
            int worker = status.MPI_SOURCE;
            active -= 1;

            bool reserve_one_for_master = master_compute && (total - next) <= 1;
            if (next < total && !reserve_one_for_master) {
                send_string(serialize_image_task(frame.tiles[next++]), worker, task_tag);
                active += 1;
            }
            continue;
        }

        if (!master_compute && next < total) {
            throw std::runtime_error("live distributed mode has unscheduled tasks but no active worker");
        }
    }
    return merge_frame_detections(cfg, frame_detections, frame.width, frame.height);
}

static std::vector<Detection> process_live_frame_anchor(
    const Config& cfg,
    DetectorRunner& detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics
) {
    ImageTask full_frame;
    full_frame.task.task_id = frame.frame_id;
    full_frame.task.frame_id = frame.frame_id;
    full_frame.task.tile_id = -1;
    full_frame.task.x1 = 0;
    full_frame.task.y1 = 0;
    full_frame.task.x2 = frame.width;
    full_frame.task.y2 = frame.height;
    full_frame.encoded_jpeg = frame.encoded_frame;

    std::vector<Detection> detections;
    auto payload = process_one_image_task_payload(cfg, detector, full_frame, 0);
    parse_payload(payload, detections, metrics);
    return merge_frame_detections(cfg, detections, frame.width, frame.height);
}

static std::vector<Detection> merge_anchor_and_tile_detections(
    const Config& cfg,
    const std::vector<Detection>& anchors,
    const std::vector<Detection>& tiles,
    int frame_width,
    int frame_height
) {
    std::vector<Detection> combined = anchors;
    for (const auto& tile_det : tiles) {
        bool covered_by_anchor = false;
        for (const auto& anchor : anchors) {
            if (duplicate_detection(cfg, tile_det, anchor, frame_width, frame_height)) {
                covered_by_anchor = true;
                break;
            }
        }
        if (!covered_by_anchor) combined.push_back(tile_det);
    }
    return merge_frame_detections(cfg, combined, frame_width, frame_height);
}

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
