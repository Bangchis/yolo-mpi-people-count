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
