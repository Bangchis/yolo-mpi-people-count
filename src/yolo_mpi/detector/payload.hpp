// Forward declaration because task processing serializes before helper definitions below.
static std::string serialize_detections(const std::vector<Detection>& detections);

// Forward declaration for the same line-oriented metrics format.
static std::string serialize_metrics(const Metrics& m);

// Run one offline task and return a text payload containing detections plus metrics.
static std::string process_one_task_payload(const Config& cfg, DetectorRunner& detector, const Task& task, int rank, double comm_ms = 0.0) {
    Metrics m;
    m.rank = rank;
    m.hostname = hostname();
    std::vector<Detection> detections;
    auto t0 = std::chrono::steady_clock::now();
    if (cfg.sleep_ms > 0) {
        int jitter = (task.frame_id + task.tile_id) % 5;
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms + jitter));
    }
    auto y0 = std::chrono::steady_clock::now();
    auto task_dets = detector.detect(task);
    auto y1 = std::chrono::steady_clock::now();
    detections.insert(detections.end(), task_dets.begin(), task_dets.end());
    auto t1 = std::chrono::steady_clock::now();
    m.tasks_done = 1;
    m.frames_done = task.tile_id == 0 ? 1 : 0;
    m.yolo_ms = std::chrono::duration<double, std::milli>(y1 - y0).count();
    m.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    m.comm_ms = comm_ms;
    return serialize_detections(detections) + serialize_metrics(m);
}

// Run one live JPEG tile task and return the same DET/MET text payload format.
static std::string process_one_image_task_payload(const Config& cfg, DetectorRunner& detector, const ImageTask& image_task, int rank) {
    Metrics m;
    m.rank = rank;
    m.hostname = hostname();
    std::vector<Detection> detections;
    auto t0 = std::chrono::steady_clock::now();
    if (cfg.sleep_ms > 0) {
        int jitter = (image_task.task.frame_id + image_task.task.tile_id) % 5;
        std::this_thread::sleep_for(std::chrono::milliseconds(cfg.sleep_ms + jitter));
    }
    auto y0 = std::chrono::steady_clock::now();
    auto task_dets = detector.detect_image(image_task);
    auto y1 = std::chrono::steady_clock::now();
    detections.insert(detections.end(), task_dets.begin(), task_dets.end());
    auto t1 = std::chrono::steady_clock::now();
    m.tasks_done = 1;
    m.frames_done = image_task.task.tile_id == 0 ? 1 : 0;
    m.yolo_ms = std::chrono::duration<double, std::milli>(y1 - y0).count();
    m.compute_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return serialize_detections(detections) + serialize_metrics(m);
}

// Encode detection rows as line-oriented text so MPI can gather variable-size results.
static std::string serialize_detections(const std::vector<Detection>& detections) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    for (const auto& det : detections) {
        out << "DET," << det.frame_id << "," << det.tile_id << "," << det.rank << ","
            << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << ","
            << det.conf << "," << det.cls << "\n";
    }
    return out.str();
}

// Encode one rank/task timing row for rank_metrics.csv.
static std::string serialize_metrics(const Metrics& m) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "MET," << m.rank << "," << m.hostname << "," << m.tasks_done << ","
        << m.frames_done << "," << m.compute_ms << "," << m.io_ms << ","
        << m.yolo_ms << "," << m.comm_ms << "," << m.idle_ms << "\n";
    return out.str();
}

// Parse one DET line back into a Detection object on rank 0.
static Detection parse_detection_line(const std::string& line) {
    std::string normalized = line;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream iss(normalized);
    std::string tag;
    Detection det;
    iss >> tag >> det.frame_id >> det.tile_id >> det.rank >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.conf >> det.cls;
    return det;
}

// Parse one MET line back into a Metrics object on rank 0.
static Metrics parse_metrics_line(const std::string& line) {
    std::string normalized = line;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream iss(normalized);
    std::string tag;
    Metrics m;
    iss >> tag >> m.rank >> m.hostname >> m.tasks_done >> m.frames_done >> m.compute_ms >> m.io_ms >> m.yolo_ms >> m.comm_ms >> m.idle_ms;
    return m;
}

// Split a mixed payload into detection rows and metrics rows.
static void parse_payload(const std::string& payload, std::vector<Detection>& detections, std::vector<Metrics>& metrics) {
    std::istringstream in(payload);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("DET,", 0) == 0) detections.push_back(parse_detection_line(line));
        else if (line.rfind("MET,", 0) == 0) metrics.push_back(parse_metrics_line(line));
    }
}

// Sum repeated task-level metric rows into one row per MPI rank.
static std::vector<Metrics> aggregate_metrics_by_rank(const std::vector<Metrics>& rows) {
    std::map<int, Metrics> grouped;
    for (const auto& row : rows) {
        auto it = grouped.find(row.rank);
        if (it == grouped.end()) {
            grouped[row.rank] = row;
            continue;
        }
        auto& dst = it->second;
        if (dst.hostname.empty()) dst.hostname = row.hostname;
        dst.tasks_done += row.tasks_done;
        dst.frames_done += row.frames_done;
        dst.compute_ms += row.compute_ms;
        dst.io_ms += row.io_ms;
        dst.yolo_ms += row.yolo_ms;
        dst.comm_ms += row.comm_ms;
        dst.idle_ms += row.idle_ms;
    }
    std::vector<Metrics> out;
    for (auto& [rank, metric] : grouped) out.push_back(metric);
    return out;
}
