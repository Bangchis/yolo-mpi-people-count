// Config parsing and task generation.
// This file answers: "what input did the user request, and what are the tasks?"

static std::string hostname() {
    std::array<char, 256> buf{};
    if (gethostname(buf.data(), buf.size() - 1) == 0) {
        return std::string(buf.data());
    }
    return "unknown";
}

static bool read_bool_arg(const std::string& value) {
    return value == "1" || value == "true" || value == "TRUE" || value == "yes";
}

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
                << "  --python .venv/bin/python --worker-script scripts/yolo_worker.py\n";
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

static std::pair<int, int> parse_tile_grid(const std::string& grid) {
    auto pos = grid.find('x');
    if (pos == std::string::npos) pos = grid.find('X');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid --tile-grid, expected COLSxROWS");
    }
    int cols = std::stoi(grid.substr(0, pos));
    int rows = std::stoi(grid.substr(pos + 1));
    if (cols <= 0 || rows <= 0) throw std::runtime_error("Tile grid dimensions must be positive");
    return {cols, rows};
}

static std::vector<Task> make_tasks(const Config& cfg) {
    auto [cols, rows] = parse_tile_grid(cfg.tile_grid);
    std::vector<Task> tasks;
    int task_id = 0;
    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        int tile_id = 0;
        for (int row = 0; row < rows; ++row) {
            int base_y1 = static_cast<int>(std::llround(row * cfg.height / static_cast<double>(rows)));
            int base_y2 = static_cast<int>(std::llround((row + 1) * cfg.height / static_cast<double>(rows)));
            for (int col = 0; col < cols; ++col) {
                int base_x1 = static_cast<int>(std::llround(col * cfg.width / static_cast<double>(cols)));
                int base_x2 = static_cast<int>(std::llround((col + 1) * cfg.width / static_cast<double>(cols)));
                Task task;
                task.task_id = task_id++;
                task.frame_id = frame;
                task.tile_id = tile_id++;
                task.x1 = std::max(0, base_x1 - cfg.overlap);
                task.y1 = std::max(0, base_y1 - cfg.overlap);
                task.x2 = std::min(cfg.width, base_x2 + cfg.overlap);
                task.y2 = std::min(cfg.height, base_y2 + cfg.overlap);
                tasks.push_back(task);
            }
        }
    }
    return tasks;
}

static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

static std::string command_for_task(std::string command, const Config& cfg, const Task& task) {
    command = replace_all(command, "{source}", cfg.source);
    command = replace_all(command, "{model}", cfg.model);
    command = replace_all(command, "{device}", cfg.device);
    command = replace_all(command, "{imgsz}", std::to_string(cfg.imgsz));
    command = replace_all(command, "{conf}", std::to_string(cfg.conf));
    command = replace_all(command, "{iou}", std::to_string(cfg.iou));
    command = replace_all(command, "{frame_id}", std::to_string(task.frame_id));
    command = replace_all(command, "{tile_id}", std::to_string(task.tile_id));
    command = replace_all(command, "{x1}", std::to_string(task.x1));
    command = replace_all(command, "{y1}", std::to_string(task.y1));
    command = replace_all(command, "{x2}", std::to_string(task.x2));
    command = replace_all(command, "{y2}", std::to_string(task.y2));
    return command;
}

static std::vector<Detection> mock_detector(const Task& task, int rank) {
    std::vector<Detection> detections;
    int w = std::max(1, task.x2 - task.x1);
    int h = std::max(1, task.y2 - task.y1);
    int count = 1 + ((task.frame_id + task.tile_id) % 3);
    for (int i = 0; i < count; ++i) {
        double cx = task.x1 + (0.22 + 0.22 * i + 0.03 * (task.frame_id % 5)) * w;
        double cy = task.y1 + (0.28 + 0.17 * i + 0.02 * (task.tile_id % 4)) * h;
        double bw = std::max(24.0, 0.12 * w);
        double bh = std::max(48.0, 0.24 * h);
        Detection det;
        det.frame_id = task.frame_id;
        det.tile_id = task.tile_id;
        det.rank = rank;
        det.x1 = std::max<double>(0, cx - bw / 2);
        det.y1 = std::max<double>(0, cy - bh / 2);
        det.x2 = cx + bw / 2;
        det.y2 = cy + bh / 2;
        det.conf = 0.92 - 0.03 * i;
        det.cls = 0;
        detections.push_back(det);
    }
    return detections;
}

static std::vector<Detection> command_detector(const Config& cfg, const Task& task, int rank) {
    std::vector<Detection> detections;
    std::string command = command_for_task(cfg.detector_command, cfg, task);
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to run detector command");
    }
    std::array<char, 512> buffer{};
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        std::string line(buffer.data());
        if (line.empty() || line[0] == '#') continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0;
        if (!(iss >> x1 >> y1 >> x2 >> y2 >> conf)) continue;
        Detection det;
        det.frame_id = task.frame_id;
        det.tile_id = task.tile_id;
        det.rank = rank;
        det.x1 = x1 + task.x1;
        det.y1 = y1 + task.y1;
        det.x2 = x2 + task.x1;
        det.y2 = y2 + task.y1;
        det.conf = conf;
        det.cls = 0;
        detections.push_back(det);
    }
    int rc = pclose(pipe);
    if (rc != 0) {
        std::cerr << "WARNING detector command returned non-zero: " << rc << "\n";
    }
    return detections;
}

static bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}
