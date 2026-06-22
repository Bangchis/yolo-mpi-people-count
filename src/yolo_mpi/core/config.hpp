static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto need_value = [&](const std::string& opt) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + opt);
            }
            return argv[++i];
        };
        if (key == "--source") cfg.source = need_value(key);
        else if (key == "--model") cfg.model = need_value(key);
        else if (key == "--device") cfg.device = need_value(key);
        else if (key == "--imgsz") cfg.imgsz = std::stoi(need_value(key));
        else if (key == "--conf") cfg.conf = std::stod(need_value(key));
        else if (key == "--iou") cfg.iou = std::stod(need_value(key));
        else if (key == "--tile-grid") cfg.tile_grid = need_value(key);
        else if (key == "--overlap") cfg.overlap = std::stoi(need_value(key));
        else if (key == "--tile-owner-filter") cfg.tile_owner_filter = read_bool_arg(need_value(key));
        else if (key == "--dedup-ios") cfg.duplicate_ios = std::stod(need_value(key));
        else if (key == "--dedup-center") cfg.duplicate_center = std::stod(need_value(key));
        else if (key == "--dedup-axis-overlap") cfg.duplicate_axis_overlap = std::stod(need_value(key));
        else if (key == "--dedup-gap") cfg.duplicate_gap = std::stod(need_value(key));
        else if (key == "--dedup-near-camera") cfg.duplicate_near_camera = read_bool_arg(need_value(key));
        else if (key == "--dedup-large-area-ratio") cfg.duplicate_large_area_ratio = std::stod(need_value(key));
        else if (key == "--dedup-merge") cfg.duplicate_merge = read_bool_arg(need_value(key));
        else if (key == "--schedule") cfg.schedule = need_value(key);
        else if (key == "--chunk-size") cfg.chunk_size = std::stoi(need_value(key));
        else if (key == "--frames") cfg.frames = std::stoi(need_value(key));
        else if (key == "--start-frame") cfg.start_frame = std::stoi(need_value(key));
        else if (key == "--width") cfg.width = std::stoi(need_value(key));
        else if (key == "--height") cfg.height = std::stoi(need_value(key));
        else if (key == "--output") cfg.output = need_value(key);
        else if (key == "--run-id") cfg.run_id = need_value(key);
        else if (key == "--master-compute") cfg.master_compute = read_bool_arg(need_value(key));
        else if (key == "--verify") cfg.verify = read_bool_arg(need_value(key));
        else if (key == "--detector") cfg.detector = need_value(key);
        else if (key == "--detector-command") cfg.detector_command = need_value(key);
        else if (key == "--python") cfg.python_bin = need_value(key);
        else if (key == "--worker-script") cfg.worker_script = need_value(key);
        else if (key == "--cpu-fallback") cfg.cpu_fallback = read_bool_arg(need_value(key));
        else if (key == "--sleep-ms") cfg.sleep_ms = std::stoi(need_value(key));
        else if (key == "--live") cfg.live = read_bool_arg(need_value(key));
        else if (key == "--camera-index") cfg.camera_index = std::stoi(need_value(key));
        else if (key == "--live-video-source") cfg.live_video_source = need_value(key);
        else if (key == "--camera-script") cfg.camera_script = need_value(key);
        else if (key == "--viewer-script") cfg.viewer_script = need_value(key);
        else if (key == "--live-view") cfg.live_view = read_bool_arg(need_value(key));
        else if (key == "--live-master-compute") cfg.live_master_compute = read_bool_arg(need_value(key));
        else if (key == "--live-anchor-full-frame") cfg.live_anchor_full_frame = read_bool_arg(need_value(key));
        else if (key == "--live-temporal-dedup") cfg.live_temporal_dedup = read_bool_arg(need_value(key));
        else if (key == "--live-temporal-center") cfg.live_temporal_center = std::stod(need_value(key));
        else if (key == "--live-temporal-ios") cfg.live_temporal_ios = std::stod(need_value(key));
        else if (key == "--live-fps") cfg.live_fps = std::stod(need_value(key));
        else if (key == "--jpeg-quality") cfg.jpeg_quality = std::stoi(need_value(key));
        else if (key == "--help") {
            if (cfg.run_id.empty()) std::cout << "";
            std::cout
                << "Usage: yolo_mpi_cpp [options]\n"
                << "  --frames N --tile-grid COLSxROWS --schedule static|dynamic\n"
                << "  --master-compute 1 lets rank 0 use the master GPU in dynamic mode\n"
                << "  --detector mock|yolo|command\n"
                << "  --live 1 --camera-index 0 --live-view 1\n"
                << "  --python .venv/bin/python --worker-script scripts/runtime/yolo_worker.py\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }
    if (cfg.frames <= 0 || cfg.width <= 0 || cfg.height <= 0 || cfg.chunk_size <= 0) {
        throw std::runtime_error("frames, width, height, and chunk-size must be positive");
    }
    if (cfg.schedule != "static" && cfg.schedule != "dynamic") {
        throw std::runtime_error("--schedule must be static or dynamic");
    }
    if (cfg.detector != "mock" && cfg.detector != "yolo" && cfg.detector != "command") {
        throw std::runtime_error("--detector must be mock, yolo, or command");
    }
    if (cfg.detector == "command" && cfg.detector_command.empty()) {
        throw std::runtime_error("--detector-command is required when --detector command");
    }
    if (cfg.live && cfg.detector == "command") {
        throw std::runtime_error("--live supports detector mock or yolo");
    }
    if (cfg.iou < 0 || cfg.iou > 1 || cfg.duplicate_ios < 0 || cfg.duplicate_ios > 1 ||
        cfg.duplicate_center < 0 || cfg.duplicate_axis_overlap < 0 || cfg.duplicate_axis_overlap > 1 ||
        cfg.duplicate_gap < 0 || cfg.duplicate_large_area_ratio < 0) {
        throw std::runtime_error("--iou, --dedup-ios, --dedup-center, --dedup-axis-overlap, --dedup-gap, and --dedup-large-area-ratio are out of range");
    }
    if (cfg.live_temporal_center < 0 || cfg.live_temporal_ios < 0 || cfg.live_temporal_ios > 1) {
        throw std::runtime_error("--live-temporal-center and --live-temporal-ios are out of range");
    }
    return cfg;
}
