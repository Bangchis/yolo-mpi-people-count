class DetectorRunner {
public:
    // Create the detector backend once per rank; YOLO keeps a long-lived Python child.
    DetectorRunner(const Config& cfg, int rank) : cfg_(cfg), rank_(rank) {
        if (cfg_.detector == "yolo") {
            // Loading the model once per MPI process avoids huge per-task overhead.
            yolo_worker_ = std::make_unique<YoloWorkerProcess>(cfg_, rank_);
        }
    }

    // Disable copying because the YOLO backend owns a child process.
    DetectorRunner(const DetectorRunner&) = delete;
    DetectorRunner& operator=(const DetectorRunner&) = delete;

    // Detect one offline frame/tile task using the selected backend.
    std::vector<Detection> detect(const Task& task) {
        if (cfg_.detector == "yolo") {
            // Real YOLO path for report/demo runs.
            return yolo_worker_->detect(task);
        }
        if (cfg_.detector == "command") {
            // Optional adapter for custom detector experiments.
            return command_detector(cfg_, task, rank_);
        }
        // Mock is the default smoke-test backend.
        return mock_detector(task, rank_);
    }

    // Detect one live JPEG tile; command detector is intentionally offline-only.
    std::vector<Detection> detect_image(const ImageTask& image_task) {
        if (cfg_.detector == "yolo") {
            // Live camera sends JPEG bytes instead of asking worker to read a video frame.
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
