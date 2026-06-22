// Detector worker bridge.
// Each MPI rank keeps one local Python YOLO worker alive over stdin/stdout.

class YoloWorkerProcess {
public:
    YoloWorkerProcess(const Config& cfg, int rank) : rank_(rank) {
        // One long-lived worker per MPI rank avoids reloading the YOLO model for
        // every frame/tile task.
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
            // Child process becomes Python. Parent keeps pipe handles below.
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
                    // Worker returns tile-local boxes; C++ remaps them to frame coordinates.
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
