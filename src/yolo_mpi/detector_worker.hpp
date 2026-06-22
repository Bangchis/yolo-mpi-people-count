// Detector worker bridge.
// Each MPI rank keeps one local Python YOLO worker alive over stdin/stdout.

class YoloWorkerProcess {
public:
    YoloWorkerProcess(const Config& cfg, int rank) : rank_(rank) {
        int to_child[2] = {-1, -1};
        int from_child[2] = {-1, -1};
        if (pipe(to_child) != 0 || pipe(from_child) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }

        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }

        if (pid_ == 0) {
            dup2(to_child[0], STDIN_FILENO);
            dup2(from_child[1], STDOUT_FILENO);
            close(to_child[0]);
            close(to_child[1]);
            close(from_child[0]);
            close(from_child[1]);

            std::vector<std::string> args = {
                cfg.python_bin,
                cfg.worker_script,
                "--source", cfg.source,
                "--model", cfg.model,
                "--device", cfg.device,
                "--imgsz", std::to_string(cfg.imgsz),
                "--conf", std::to_string(cfg.conf),
                "--iou", std::to_string(cfg.iou),
                "--class-id", "0",
                "--cpu-fallback", cfg.cpu_fallback ? "1" : "0",
            };
            std::vector<char*> argv;
            argv.reserve(args.size() + 1);
            for (auto& arg : args) argv.push_back(arg.data());
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::cerr << "rank=" << rank_ << " failed to exec YOLO worker: " << std::strerror(errno) << "\n";
            _exit(127);
        }

        close(to_child[0]);
        close(from_child[1]);
        input_ = fdopen(to_child[1], "w");
        output_ = fdopen(from_child[0], "r");
        if (!input_ || !output_) {
            throw std::runtime_error("fdopen failed for YOLO worker pipes");
        }
        setvbuf(input_, nullptr, _IOLBF, 0);
        wait_until_ready();
    }

    YoloWorkerProcess(const YoloWorkerProcess&) = delete;
    YoloWorkerProcess& operator=(const YoloWorkerProcess&) = delete;

    ~YoloWorkerProcess() {
        if (input_) {
            std::fprintf(input_, "QUIT\n");
            std::fflush(input_);
        }
        if (output_) {
            std::string line;
            read_line_no_throw(line);
        }
        if (input_) {
            std::fclose(input_);
            input_ = nullptr;
        }
        if (output_) {
            std::fclose(output_);
            output_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    std::vector<Detection> detect(const Task& task) {
        if (!input_ || !output_) {
            throw std::runtime_error("YOLO worker is not running");
        }
        std::fprintf(
            input_,
            "TASK %d %d %d %d %d %d %d\n",
            task.task_id,
            task.frame_id,
            task.tile_id,
            task.x1,
            task.y1,
            task.x2,
            task.y2
        );
        std::fflush(input_);
        return read_detection_response(task);
    }

    std::vector<Detection> detect_image(const Task& task, const std::string& encoded_jpeg) {
        if (!input_ || !output_) {
            throw std::runtime_error("YOLO worker is not running");
        }
        std::fprintf(
            input_,
            "IMAGE %d %d %d %d %d %d %d ",
            task.task_id,
            task.frame_id,
            task.tile_id,
            task.x1,
            task.y1,
            task.x2,
            task.y2
        );
        std::fwrite(encoded_jpeg.data(), 1, encoded_jpeg.size(), input_);
        std::fprintf(input_, "\n");
        std::fflush(input_);
        return read_detection_response(task);
    }

private:
    std::vector<Detection> read_detection_response(const Task& task) {
        std::vector<Detection> detections;
        std::string line;
        while (read_line(line)) {
            if (starts_with(line, "ERROR ")) {
                throw std::runtime_error("YOLO worker error: " + line);
            }
            if (starts_with(line, "BEGIN ")) {
                continue;
            }
            if (starts_with(line, "DET ")) {
                std::istringstream iss(line);
                std::string tag;
                double x1 = 0, y1 = 0, x2 = 0, y2 = 0, conf = 0;
                if (iss >> tag >> x1 >> y1 >> x2 >> y2 >> conf) {
                    Detection det;
                    det.frame_id = task.frame_id;
                    det.tile_id = task.tile_id;
                    det.rank = rank_;
                    det.x1 = x1 + task.x1;
                    det.y1 = y1 + task.y1;
                    det.x2 = x2 + task.x1;
                    det.y2 = y2 + task.y1;
                    det.conf = conf;
                    det.cls = 0;
                    detections.push_back(det);
                }
                continue;
            }
            if (starts_with(line, "END ")) {
                return detections;
            }
        }
        throw std::runtime_error("YOLO worker exited before END");
    }

    void wait_until_ready() {
        std::string line;
        while (read_line(line)) {
            if (starts_with(line, "READY ")) {
                std::cerr << "rank=" << rank_ << " " << line << "\n";
                return;
            }
            if (starts_with(line, "ERROR ")) {
                throw std::runtime_error("YOLO worker startup failed: " + line);
            }
            std::cerr << "rank=" << rank_ << " YOLO_WORKER: " << line << "\n";
        }
        throw std::runtime_error("YOLO worker exited before READY");
    }

    bool read_line(std::string& line) {
        std::array<char, 4096> buffer{};
        if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), output_) == nullptr) {
            return false;
        }
        line = buffer.data();
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        return true;
    }

    bool read_line_no_throw(std::string& line) {
        if (!output_) return false;
        return read_line(line);
    }

    int rank_ = 0;
    pid_t pid_ = -1;
    FILE* input_ = nullptr;
    FILE* output_ = nullptr;
};

class DetectorRunner {
public:
    DetectorRunner(const Config& cfg, int rank) : cfg_(cfg), rank_(rank) {
        if (cfg_.detector == "yolo") {
            yolo_worker_ = std::make_unique<YoloWorkerProcess>(cfg_, rank_);
        }
    }

    std::vector<Detection> detect(const Task& task) {
        if (cfg_.detector == "yolo") {
            return yolo_worker_->detect(task);
        }
        if (cfg_.detector == "command") {
            return command_detector(cfg_, task, rank_);
        }
        return mock_detector(task, rank_);
    }

    std::vector<Detection> detect_image(const ImageTask& image_task) {
        if (cfg_.detector == "yolo") {
            return yolo_worker_->detect_image(image_task.task, image_task.encoded_jpeg);
        }
        if (cfg_.detector == "command") {
            throw std::runtime_error("command detector is not supported for live image tasks");
        }
        return mock_detector(image_task.task, rank_);
    }

private:
    const Config& cfg_;
    int rank_;
    std::unique_ptr<YoloWorkerProcess> yolo_worker_;
};

static std::string serialize_detections(const std::vector<Detection>& detections);
static std::string serialize_metrics(const Metrics& m);

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

static std::string serialize_metrics(const Metrics& m) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    out << "MET," << m.rank << "," << m.hostname << "," << m.tasks_done << ","
        << m.frames_done << "," << m.compute_ms << "," << m.io_ms << ","
        << m.yolo_ms << "," << m.comm_ms << "," << m.idle_ms << "\n";
    return out.str();
}

static Detection parse_detection_line(const std::string& line) {
    std::string normalized = line;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream iss(normalized);
    std::string tag;
    Detection det;
    iss >> tag >> det.frame_id >> det.tile_id >> det.rank >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.conf >> det.cls;
    return det;
}

static Metrics parse_metrics_line(const std::string& line) {
    std::string normalized = line;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');
    std::istringstream iss(normalized);
    std::string tag;
    Metrics m;
    iss >> tag >> m.rank >> m.hostname >> m.tasks_done >> m.frames_done >> m.compute_ms >> m.io_ms >> m.yolo_ms >> m.comm_ms >> m.idle_ms;
    return m;
}

static void parse_payload(const std::string& payload, std::vector<Detection>& detections, std::vector<Metrics>& metrics) {
    std::istringstream in(payload);
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("DET,", 0) == 0) detections.push_back(parse_detection_line(line));
        else if (line.rfind("MET,", 0) == 0) metrics.push_back(parse_metrics_line(line));
    }
}

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
