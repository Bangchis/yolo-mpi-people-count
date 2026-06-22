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

