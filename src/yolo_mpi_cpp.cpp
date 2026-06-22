#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

struct Config {
    std::string source = "data/classroom.mp4";
    std::string model = "yolo11n.pt";
    std::string device = "mps";
    int imgsz = 512;
    double conf = 0.35;
    double iou = 0.50;
    std::string tile_grid = "1x1";
    int overlap = 64;
    bool tile_owner_filter = true;
    double duplicate_ios = 0.70;
    double duplicate_center = 0.30;
    double duplicate_axis_overlap = 0.70;
    double duplicate_gap = 0.08;
    bool duplicate_near_camera = false;
    double duplicate_large_area_ratio = 0.12;
    bool duplicate_merge = true;
    std::string schedule = "static";
    int chunk_size = 1;
    int frames = 20;
    int start_frame = 0;
    int width = 1280;
    int height = 720;
    std::string output = "results/cpp_run";
    std::string run_id = "cpp_run";
    bool master_compute = true;
    bool verify = false;
    bool write_video = false;
    std::string detector = "mock";
    std::string detector_command;
    std::string python_bin = ".venv/bin/python";
    std::string worker_script = "scripts/yolo_worker.py";
    bool cpu_fallback = true;
    int sleep_ms = 0;
    bool live = false;
    int camera_index = 0;
    std::string live_video_source;
    std::string camera_script = "scripts/camera_tile_source.py";
    std::string viewer_script = "scripts/live_viewer.py";
    bool live_view = false;
    bool live_master_compute = true;
    bool live_anchor_full_frame = false;
    bool live_temporal_dedup = false;
    double live_temporal_center = 0.55;
    double live_temporal_ios = 0.35;
    double live_fps = 0.0;
    int jpeg_quality = 80;
};

struct Task {
    int task_id = 0;
    int frame_id = 0;
    int tile_id = 0;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

struct Detection {
    int frame_id = 0;
    int tile_id = 0;
    int rank = 0;
    double x1 = 0;
    double y1 = 0;
    double x2 = 0;
    double y2 = 0;
    double conf = 0;
    int cls = 0;
};

struct Metrics {
    int rank = 0;
    std::string hostname;
    int tasks_done = 0;
    int frames_done = 0;
    double compute_ms = 0;
    double io_ms = 0;
    double yolo_ms = 0;
    double comm_ms = 0;
    double idle_ms = 0;
};

struct ImageTask {
    Task task;
    std::string encoded_jpeg;
};

struct CameraFrame {
    int frame_id = 0;
    int width = 0;
    int height = 0;
    double capture_ms = 0;
    std::string encoded_frame;
    std::vector<ImageTask> tiles;
};

struct LiveFrameEvent {
    int frame_id = 0;
    int person_count = 0;
    int tasks = 0;
    double capture_ms = 0;
    double latency_ms = 0;
};

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
        else if (key == "--write-video") cfg.write_video = read_bool_arg(need_value(key));
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

static std::string gather_string(const std::string& local, int root, MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int local_size = static_cast<int>(local.size());
    std::vector<int> sizes(size);
    MPI_Gather(&local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, root, comm);
    std::vector<int> displs(size, 0);
    int total = 0;
    if (rank == root) {
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += sizes[i];
        }
    }
    std::string gathered;
    if (rank == root) gathered.resize(total);
    MPI_Gatherv(
        local.data(),
        local_size,
        MPI_CHAR,
        rank == root ? gathered.data() : nullptr,
        sizes.data(),
        displs.data(),
        MPI_CHAR,
        root,
        comm
    );
    return gathered;
}

static std::vector<Task> select_static_tasks(const std::vector<Task>& tasks, int rank, int world_size, int chunk_size) {
    std::vector<Task> selected;
    for (const auto& task : tasks) {
        int block_id = task.task_id / std::max(1, chunk_size);
        if (block_id % world_size == rank) selected.push_back(task);
    }
    return selected;
}

static std::string process_tasks_payload(
    const Config& cfg,
    const std::vector<Task>& tasks,
    int rank,
    Metrics* out_metrics = nullptr,
    bool include_metrics = true
) {
    Metrics m;
    m.rank = rank;
    m.hostname = hostname();
    std::vector<Detection> detections;
    DetectorRunner detector(cfg, rank);
    for (const auto& task : tasks) {
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
        m.tasks_done += 1;
        if (task.tile_id == 0) m.frames_done += 1;
        m.yolo_ms += std::chrono::duration<double, std::milli>(y1 - y0).count();
        m.compute_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    if (out_metrics) *out_metrics = m;
    std::string payload = serialize_detections(detections);
    if (include_metrics) payload += serialize_metrics(m);
    return payload;
}

static std::string run_static(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    int rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    auto local_tasks = select_static_tasks(tasks, rank, world_size, cfg.chunk_size);
    Metrics local_metrics;
    auto payload = process_tasks_payload(cfg, local_tasks, rank, &local_metrics, false);
    auto c0 = std::chrono::steady_clock::now();
    std::string gathered = gather_string(payload, 0, comm);
    auto c1 = std::chrono::steady_clock::now();
    local_metrics.comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
    std::string gathered_metrics = gather_string(serialize_metrics(local_metrics), 0, comm);
    if (rank == 0) return gathered + gathered_metrics;
    return "";
}

static void send_task(const Task& task, int dest) {
    int raw[7] = {task.task_id, task.frame_id, task.tile_id, task.x1, task.y1, task.x2, task.y2};
    MPI_Send(raw, 7, MPI_INT, dest, 10, MPI_COMM_WORLD);
}

static Task recv_task(int* source_tag = nullptr) {
    MPI_Status status;
    int raw[7] = {};
    MPI_Recv(raw, 7, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    if (source_tag) *source_tag = status.MPI_TAG;
    Task task;
    task.task_id = raw[0];
    task.frame_id = raw[1];
    task.tile_id = raw[2];
    task.x1 = raw[3];
    task.y1 = raw[4];
    task.x2 = raw[5];
    task.y2 = raw[6];
    return task;
}

static void send_string(const std::string& payload, int dest, int tag) {
    int n = static_cast<int>(payload.size());
    MPI_Send(&n, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    if (n > 0) MPI_Send(payload.data(), n, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
}

static std::string recv_string(int source, int tag, MPI_Status* status_out = nullptr) {
    MPI_Status status;
    int n = 0;
    MPI_Recv(&n, 1, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
    std::string payload;
    payload.resize(n);
    if (n > 0) MPI_Recv(payload.data(), n, MPI_CHAR, status.MPI_SOURCE, tag, MPI_COMM_WORLD, &status);
    if (status_out) *status_out = status;
    return payload;
}

static bool read_pipe_line(FILE* stream, std::string& line) {
    char* buffer = nullptr;
    size_t cap = 0;
    ssize_t n = getline(&buffer, &cap, stream);
    if (n < 0) {
        std::free(buffer);
        return false;
    }
    line.assign(buffer, static_cast<size_t>(n));
    std::free(buffer);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    return true;
}

class OutputPipeProcess {
public:
    explicit OutputPipeProcess(const std::vector<std::string>& args) {
        int from_child[2] = {-1, -1};
        if (pipe(from_child) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }
        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }
        if (pid_ == 0) {
            dup2(from_child[1], STDOUT_FILENO);
            close(from_child[0]);
            close(from_child[1]);
            std::vector<std::string> mutable_args = args;
            std::vector<char*> argv;
            argv.reserve(mutable_args.size() + 1);
            for (auto& arg : mutable_args) argv.push_back(arg.data());
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::cerr << "failed to exec " << argv[0] << ": " << std::strerror(errno) << "\n";
            _exit(127);
        }
        close(from_child[1]);
        output_ = fdopen(from_child[0], "r");
        if (!output_) {
            throw std::runtime_error("fdopen failed for output pipe");
        }
    }

    OutputPipeProcess(const OutputPipeProcess&) = delete;
    OutputPipeProcess& operator=(const OutputPipeProcess&) = delete;

    ~OutputPipeProcess() {
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

    bool read_line(std::string& line) {
        if (!output_) return false;
        return read_pipe_line(output_, line);
    }

private:
    pid_t pid_ = -1;
    FILE* output_ = nullptr;
};

class InputPipeProcess {
public:
    explicit InputPipeProcess(const std::vector<std::string>& args) {
        int to_child[2] = {-1, -1};
        if (pipe(to_child) != 0) {
            throw std::runtime_error(std::string("pipe failed: ") + std::strerror(errno));
        }
        pid_ = fork();
        if (pid_ < 0) {
            throw std::runtime_error(std::string("fork failed: ") + std::strerror(errno));
        }
        if (pid_ == 0) {
            dup2(to_child[0], STDIN_FILENO);
            close(to_child[0]);
            close(to_child[1]);
            std::vector<std::string> mutable_args = args;
            std::vector<char*> argv;
            argv.reserve(mutable_args.size() + 1);
            for (auto& arg : mutable_args) argv.push_back(arg.data());
            argv.push_back(nullptr);
            execvp(argv[0], argv.data());
            std::cerr << "failed to exec " << argv[0] << ": " << std::strerror(errno) << "\n";
            _exit(127);
        }
        close(to_child[0]);
        input_ = fdopen(to_child[1], "w");
        if (!input_) {
            throw std::runtime_error("fdopen failed for input pipe");
        }
        setvbuf(input_, nullptr, _IOLBF, 0);
    }

    InputPipeProcess(const InputPipeProcess&) = delete;
    InputPipeProcess& operator=(const InputPipeProcess&) = delete;

    ~InputPipeProcess() {
        if (input_) {
            write_line("QUIT");
            std::fclose(input_);
            input_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

    void write_line(const std::string& line) {
        if (!input_) return;
        std::fwrite(line.data(), 1, line.size(), input_);
        std::fwrite("\n", 1, 1, input_);
        std::fflush(input_);
    }

private:
    pid_t pid_ = -1;
    FILE* input_ = nullptr;
};

static std::string run_dynamic(const Config& cfg, const std::vector<Task>& tasks, MPI_Comm comm) {
    int rank = 0, world_size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);
    if (world_size == 1) return run_static(cfg, tasks, comm);

    const int stop_tag = 30;
    const int result_tag = 20;

    if (rank == 0) {
        int next = 0;
        int completed = 0;
        int stopped = 0;
        int active = 0;
        double master_comm_ms = 0.0;
        const int total = static_cast<int>(tasks.size());
        const bool master_compute = cfg.master_compute;
        const int initial_worker_tasks = master_compute
            ? std::min(world_size - 1, std::max(0, total - 1))
            : std::min(world_size - 1, total);

        for (int worker = 1; worker < world_size; ++worker) {
            if (next < initial_worker_tasks) {
                auto c0 = std::chrono::steady_clock::now();
                send_task(tasks[next++], worker);
                auto c1 = std::chrono::steady_clock::now();
                master_comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
                active += 1;
            } else {
                int empty[7] = {};
                auto c0 = std::chrono::steady_clock::now();
                MPI_Send(empty, 7, MPI_INT, worker, stop_tag, MPI_COMM_WORLD);
                auto c1 = std::chrono::steady_clock::now();
                master_comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
                stopped += 1;
            }
        }

        auto receive_worker_result = [&](bool block, std::ostringstream& all) -> bool {
            if (active <= 0) return false;
            if (!block) {
                int flag = 0;
                MPI_Status probe_status;
                MPI_Iprobe(MPI_ANY_SOURCE, result_tag, MPI_COMM_WORLD, &flag, &probe_status);
                if (!flag) return false;
            }
            MPI_Status status;
            auto c0 = std::chrono::steady_clock::now();
            auto payload = recv_string(MPI_ANY_SOURCE, result_tag, &status);
            auto c1 = std::chrono::steady_clock::now();
            master_comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
            all << payload;
            completed += 1;
            active -= 1;
            int worker = status.MPI_SOURCE;

            bool reserve_for_master = master_compute && (total - next) <= 1;
            if (next < total && !reserve_for_master) {
                auto s0 = std::chrono::steady_clock::now();
                send_task(tasks[next++], worker);
                auto s1 = std::chrono::steady_clock::now();
                master_comm_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
                active += 1;
            } else {
                int empty[7] = {};
                auto s0 = std::chrono::steady_clock::now();
                MPI_Send(empty, 7, MPI_INT, worker, stop_tag, MPI_COMM_WORLD);
                auto s1 = std::chrono::steady_clock::now();
                master_comm_ms += std::chrono::duration<double, std::milli>(s1 - s0).count();
                stopped += 1;
            }
            return true;
        };

        std::ostringstream all;
        std::unique_ptr<DetectorRunner> master_detector;
        if (master_compute) {
            master_detector = std::make_unique<DetectorRunner>(cfg, 0);
        } else {
            all << serialize_metrics(Metrics{0, hostname(), 0, 0, 0, 0, 0, 0, 0});
        }

        while (completed < total) {
            while (receive_worker_result(false, all)) {
            }
            if (master_detector && next < total) {
                all << process_one_task_payload(cfg, *master_detector, tasks[next++], 0);
                completed += 1;
                continue;
            }
            if (receive_worker_result(true, all)) {
                continue;
            }
            if (!master_compute && next < total && active == 0) {
                throw std::runtime_error("dynamic scheduler has unscheduled tasks but no active worker");
            }
        }

        while (stopped < world_size - 1) {
            if (!receive_worker_result(true, all)) break;
        }
        all << serialize_metrics(Metrics{0, hostname(), 0, 0, 0, 0, 0, master_comm_ms, 0});
        return all.str();
    }

    DetectorRunner detector(cfg, rank);
    double pending_comm_ms = 0.0;
    while (true) {
        int tag = 0;
        auto c0 = std::chrono::steady_clock::now();
        Task task = recv_task(&tag);
        auto c1 = std::chrono::steady_clock::now();
        pending_comm_ms += std::chrono::duration<double, std::milli>(c1 - c0).count();
        if (tag == stop_tag) break;
        auto payload = process_one_task_payload(cfg, detector, task, rank, pending_comm_ms);
        pending_comm_ms = 0.0;
        send_string(payload, 0, result_tag);
    }
    return "";
}

static double box_area(const Detection& d) {
    return std::max(0.0, d.x2 - d.x1) * std::max(0.0, d.y2 - d.y1);
}

static double box_width(const Detection& d) {
    return std::max(0.0, d.x2 - d.x1);
}

static double box_height(const Detection& d) {
    return std::max(0.0, d.y2 - d.y1);
}

static double intersection_area(const Detection& a, const Detection& b) {
    double ix1 = std::max(a.x1, b.x1);
    double iy1 = std::max(a.y1, b.y1);
    double ix2 = std::min(a.x2, b.x2);
    double iy2 = std::min(a.y2, b.y2);
    double iw = std::max(0.0, ix2 - ix1);
    double ih = std::max(0.0, iy2 - iy1);
    return iw * ih;
}

static double iou(const Detection& a, const Detection& b) {
    double inter = intersection_area(a, b);
    double area_a = box_area(a);
    double area_b = box_area(b);
    double denom = area_a + area_b - inter;
    return denom > 0 ? inter / denom : 0.0;
}

static double intersection_over_smaller(const Detection& a, const Detection& b) {
    double smaller = std::min(box_area(a), box_area(b));
    if (smaller <= 0) return 0.0;
    return intersection_area(a, b) / smaller;
}

static double axis_overlap_ratio(double a1, double a2, double b1, double b2) {
    double overlap = std::max(0.0, std::min(a2, b2) - std::max(a1, b1));
    double smaller = std::min(std::max(0.0, a2 - a1), std::max(0.0, b2 - b1));
    return smaller > 0 ? overlap / smaller : 0.0;
}

static double axis_gap_ratio(double a1, double a2, double b1, double b2, double scale) {
    double gap = std::max(0.0, std::max(a1, b1) - std::min(a2, b2));
    return scale > 0 ? gap / scale : 0.0;
}

static bool close_cross_tile_duplicate(const Config& cfg, const Detection& a, const Detection& b, int frame_width, int frame_height) {
    if (!cfg.duplicate_near_camera) return false;
    if (a.tile_id == b.tile_id) return false;
    double frame_area = static_cast<double>(std::max(1, frame_width)) * static_cast<double>(std::max(1, frame_height));

    double aw = box_width(a), bw = box_width(b);
    double ah = box_height(a), bh = box_height(b);
    double min_w = std::min(aw, bw);
    double min_h = std::min(ah, bh);
    double max_w = std::max(aw, bw);
    double max_h = std::max(ah, bh);
    if (min_w <= 0 || min_h <= 0 || max_w <= 0 || max_h <= 0) return false;

    bool near_camera_scale =
        std::max(box_area(a), box_area(b)) / frame_area >= cfg.duplicate_large_area_ratio ||
        max_w / std::max(1.0, static_cast<double>(frame_width)) >= 0.35 ||
        max_h / std::max(1.0, static_cast<double>(frame_height)) >= 0.45;
    if (!near_camera_scale) return false;

    double ax = (a.x1 + a.x2) * 0.5;
    double ay = (a.y1 + a.y2) * 0.5;
    double bx = (b.x1 + b.x2) * 0.5;
    double by = (b.y1 + b.y2) * 0.5;
    double ndx = std::abs(ax - bx) / max_w;
    double ndy = std::abs(ay - by) / max_h;

    double ox = axis_overlap_ratio(a.x1, a.x2, b.x1, b.x2);
    double oy = axis_overlap_ratio(a.y1, a.y2, b.y1, b.y2);
    double gap_x = axis_gap_ratio(a.x1, a.x2, b.x1, b.x2, min_w);
    double gap_y = axis_gap_ratio(a.y1, a.y2, b.y1, b.y2, min_h);

    bool vertical_split =
        oy >= cfg.duplicate_axis_overlap &&
        gap_x <= cfg.duplicate_gap &&
        ndy <= cfg.duplicate_center;
    bool horizontal_split =
        ox >= cfg.duplicate_axis_overlap &&
        gap_y <= cfg.duplicate_gap &&
        ndx <= cfg.duplicate_center;
    bool center_close =
        ndx <= cfg.duplicate_center &&
        ndy <= cfg.duplicate_center &&
        (ox > 0.05 || oy > 0.05 || gap_x <= cfg.duplicate_gap || gap_y <= cfg.duplicate_gap);

    return vertical_split || horizontal_split || center_close;
}

static bool duplicate_detection(const Config& cfg, const Detection& candidate, const Detection& selected, int frame_width, int frame_height) {
    if (candidate.cls != selected.cls) return false;
    if (iou(candidate, selected) >= cfg.iou) return true;
    if (cfg.duplicate_ios > 0 && intersection_over_smaller(candidate, selected) >= cfg.duplicate_ios) return true;
    if (close_cross_tile_duplicate(cfg, candidate, selected, frame_width, frame_height)) return true;
    return false;
}

static Detection merge_duplicate_detection(const Detection& a, const Detection& b) {
    Detection merged = a.conf >= b.conf ? a : b;
    merged.x1 = std::min(a.x1, b.x1);
    merged.y1 = std::min(a.y1, b.y1);
    merged.x2 = std::max(a.x2, b.x2);
    merged.y2 = std::max(a.y2, b.y2);
    merged.conf = std::max(a.conf, b.conf);
    return merged;
}

static bool detection_center_inside_tile_core(const Config& cfg, const Detection& det, int frame_width, int frame_height) {
    if (!cfg.tile_owner_filter) return true;
    auto [cols, rows] = parse_tile_grid(cfg.tile_grid);
    if (cols * rows <= 1) return true;
    if (det.tile_id < 0 || det.tile_id >= cols * rows) return true;

    int col = det.tile_id % cols;
    int row = det.tile_id / cols;
    double core_x1 = std::llround(col * frame_width / static_cast<double>(cols));
    double core_x2 = std::llround((col + 1) * frame_width / static_cast<double>(cols));
    double core_y1 = std::llround(row * frame_height / static_cast<double>(rows));
    double core_y2 = std::llround((row + 1) * frame_height / static_cast<double>(rows));
    double cx = (det.x1 + det.x2) * 0.5;
    double cy = (det.y1 + det.y2) * 0.5;
    const double eps = 1e-6;
    bool x_ok = cx + eps >= core_x1 && (col == cols - 1 ? cx <= core_x2 + eps : cx < core_x2 - eps);
    bool y_ok = cy + eps >= core_y1 && (row == rows - 1 ? cy <= core_y2 + eps : cy < core_y2 - eps);
    return x_ok && y_ok;
}

static std::vector<Detection> apply_tile_owner_filter(const Config& cfg, const std::vector<Detection>& detections, int frame_width, int frame_height) {
    std::vector<Detection> filtered;
    filtered.reserve(detections.size());
    for (const auto& det : detections) {
        if (detection_center_inside_tile_core(cfg, det, frame_width, frame_height)) {
            filtered.push_back(det);
        }
    }
    return filtered;
}

static std::vector<Detection> nms(const Config& cfg, std::vector<Detection> detections, int frame_width, int frame_height) {
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.conf > b.conf;
    });
    std::vector<Detection> kept;
    for (const auto& candidate : detections) {
        bool keep = true;
        for (auto& selected : kept) {
            if (duplicate_detection(cfg, candidate, selected, frame_width, frame_height)) {
                if (cfg.duplicate_merge) {
                    selected = merge_duplicate_detection(selected, candidate);
                }
                keep = false;
                break;
            }
        }
        if (keep) kept.push_back(candidate);
    }
    return kept;
}

static bool temporal_match(const Config& cfg, const Detection& current, const Detection& previous) {
    if (current.cls != previous.cls) return false;
    if (iou(current, previous) >= 0.15) return true;
    if (intersection_over_smaller(current, previous) >= cfg.live_temporal_ios) return true;

    double cw = box_width(current), pw = box_width(previous);
    double ch = box_height(current), ph = box_height(previous);
    double max_w = std::max(cw, pw);
    double max_h = std::max(ch, ph);
    if (max_w <= 0 || max_h <= 0) return false;

    double cx = (current.x1 + current.x2) * 0.5;
    double cy = (current.y1 + current.y2) * 0.5;
    double px = (previous.x1 + previous.x2) * 0.5;
    double py = (previous.y1 + previous.y2) * 0.5;
    double ndx = std::abs(cx - px) / max_w;
    double ndy = std::abs(cy - py) / max_h;
    double ox = axis_overlap_ratio(current.x1, current.x2, previous.x1, previous.x2);
    double oy = axis_overlap_ratio(current.y1, current.y2, previous.y1, previous.y2);

    return ndx <= cfg.live_temporal_center &&
           ndy <= cfg.live_temporal_center &&
           (ox > 0.10 || oy > 0.10);
}

static std::vector<Detection> temporal_dedup_against_previous(
    const Config& cfg,
    const std::vector<Detection>& current,
    const std::vector<Detection>& previous,
    int frame_width,
    int frame_height
) {
    if (!cfg.live_temporal_dedup || current.size() < 2 || previous.empty()) return current;

    std::vector<bool> used(current.size(), false);
    std::vector<Detection> out;
    out.reserve(current.size());

    for (const auto& prev : previous) {
        std::vector<size_t> matches;
        for (size_t i = 0; i < current.size(); ++i) {
            if (!used[i] && temporal_match(cfg, current[i], prev)) {
                matches.push_back(i);
            }
        }
        if (matches.size() < 2) continue;

        Detection merged = current[matches.front()];
        used[matches.front()] = true;
        for (size_t j = 1; j < matches.size(); ++j) {
            merged = merge_duplicate_detection(merged, current[matches[j]]);
            used[matches[j]] = true;
        }
        out.push_back(merged);
    }

    for (size_t i = 0; i < current.size(); ++i) {
        if (!used[i]) out.push_back(current[i]);
    }
    return nms(cfg, out, frame_width, frame_height);
}

static std::vector<Detection> merge_frame_detections(
    const Config& cfg,
    const std::vector<Detection>& detections,
    int frame_width,
    int frame_height
) {
    auto filtered = apply_tile_owner_filter(cfg, detections, frame_width, frame_height);
    return nms(cfg, filtered, frame_width, frame_height);
}

static std::map<int, std::vector<Detection>> nms_by_frame(const Config& cfg, const std::vector<Detection>& detections) {
    std::map<int, std::vector<Detection>> grouped;
    for (const auto& det : detections) grouped[det.frame_id].push_back(det);
    for (auto& [frame, dets] : grouped) {
        (void)frame;
        dets = merge_frame_detections(cfg, dets, cfg.width, cfg.height);
    }
    return grouped;
}

static void write_frame_counts(const fs::path& path, const Config& cfg, const std::map<int, std::vector<Detection>>& by_frame) {
    std::ofstream f(path);
    f << "frame_id,person_count\n";
    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        auto it = by_frame.find(frame);
        f << frame << "," << (it == by_frame.end() ? 0 : it->second.size()) << "\n";
    }
}

static void write_bboxes(const fs::path& path, const std::map<int, std::vector<Detection>>& by_frame) {
    std::ofstream f(path);
    f << "frame_id,tile_id,rank,x1,y1,x2,y2,conf,cls\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& [frame, detections] : by_frame) {
        for (const auto& det : detections) {
            f << det.frame_id << "," << det.tile_id << "," << det.rank << ","
              << det.x1 << "," << det.y1 << "," << det.x2 << "," << det.y2 << ","
              << det.conf << "," << det.cls << "\n";
        }
    }
}

static void write_rank_metrics(const fs::path& path, std::vector<Metrics> metrics) {
    double max_compute = 0;
    for (const auto& m : metrics) max_compute = std::max(max_compute, m.compute_ms);
    for (auto& m : metrics) m.idle_ms = std::max(0.0, max_compute - m.compute_ms);
    std::sort(metrics.begin(), metrics.end(), [](const Metrics& a, const Metrics& b) { return a.rank < b.rank; });

    std::ofstream f(path);
    f << "rank,hostname,tasks_done,frames_done,compute_ms,io_ms,yolo_ms,comm_ms,idle_ms\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& m : metrics) {
        f << m.rank << "," << m.hostname << "," << m.tasks_done << "," << m.frames_done << ","
          << m.compute_ms << "," << m.io_ms << "," << m.yolo_ms << "," << m.comm_ms << "," << m.idle_ms << "\n";
    }
}

static double load_imbalance(const std::vector<Metrics>& metrics) {
    std::vector<double> active;
    for (const auto& m : metrics) {
        if (m.tasks_done > 0) active.push_back(m.compute_ms);
    }
    if (active.empty()) return 0;
    double avg = std::accumulate(active.begin(), active.end(), 0.0) / active.size();
    if (avg <= 0) return 0;
    return *std::max_element(active.begin(), active.end()) / avg;
}

static bool verify_counts(const Config& cfg, const std::vector<Task>& tasks, const std::map<int, std::vector<Detection>>& parallel_by_frame, fs::path report) {
    std::vector<Detection> serial;
    DetectorRunner detector(cfg, -1);
    for (const auto& task : tasks) {
        auto dets = detector.detect(task);
        serial.insert(serial.end(), dets.begin(), dets.end());
    }
    auto serial_by_frame = nms_by_frame(cfg, serial);
    int max_error = 0;
    double sum_error = 0;
    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        int p = parallel_by_frame.count(frame) ? static_cast<int>(parallel_by_frame.at(frame).size()) : 0;
        int s = serial_by_frame.count(frame) ? static_cast<int>(serial_by_frame.at(frame).size()) : 0;
        int err = std::abs(p - s);
        max_error = std::max(max_error, err);
        sum_error += err;
    }
    bool ok = max_error == 0;
    std::ofstream f(report);
    f << "CORRECTNESS_REPORT\n";
    f << "CORRECTNESS_PASS=" << (ok ? "YES" : "NO") << "\n";
    f << "max_count_error=" << max_error << "\n";
    f << "mean_count_error=" << (sum_error / std::max(1, cfg.frames)) << "\n";
    f << "frames_checked=" << cfg.frames << "\n";
    return ok;
}

static void write_summary(
    const fs::path& path,
    const Config& cfg,
    int world_size,
    int num_tasks,
    double total_ms,
    const std::vector<Metrics>& metrics,
    const std::map<int, std::vector<Detection>>& by_frame,
    int correctness
) {
    double compute_max = 0, compute_sum = 0, comm_sum = 0, io_sum = 0, yolo_sum = 0;
    for (const auto& m : metrics) {
        compute_max = std::max(compute_max, m.compute_ms);
    }
    double idle_sum = 0;
    for (const auto& m : metrics) {
        compute_sum += m.compute_ms;
        comm_sum += m.comm_ms;
        io_sum += m.io_ms;
        yolo_sum += m.yolo_ms;
        idle_sum += std::max(0.0, compute_max - m.compute_ms);
    }
    double avg_count = 0;
    for (int frame = cfg.start_frame; frame < cfg.start_frame + cfg.frames; ++frame) {
        auto it = by_frame.find(frame);
        avg_count += (it == by_frame.end() ? 0 : it->second.size());
    }
    avg_count /= std::max(1, cfg.frames);

    std::ofstream f(path);
    f << "run_id,language,detector,model,device,imgsz,frames,tile_grid,num_tasks,overlap,tile_owner_filter,dedup_ios,dedup_center,dedup_axis_overlap,dedup_gap,dedup_near_camera,dedup_large_area_ratio,dedup_merge,schedule,chunk_size,master_compute,world_size,video_width,video_height,total_ms_with_comm,total_ms_without_comm,compute_ms_max,compute_ms_avg,comm_ms_total,io_ms_total,yolo_ms_total,idle_ms_total,load_imbalance,avg_count,correctness_pass\n";
    f << std::fixed << std::setprecision(4);
    f << cfg.run_id << ",C++17/OpenMPI," << cfg.detector << "," << cfg.model << "," << cfg.device << ","
      << cfg.imgsz << "," << cfg.frames << "," << cfg.tile_grid << "," << num_tasks << ","
      << cfg.overlap << "," << (cfg.tile_owner_filter ? 1 : 0) << "," << cfg.duplicate_ios << ","
      << cfg.duplicate_center << "," << cfg.duplicate_axis_overlap << "," << cfg.duplicate_gap << ","
      << (cfg.duplicate_near_camera ? 1 : 0) << "," << cfg.duplicate_large_area_ratio << ","
      << (cfg.duplicate_merge ? 1 : 0) << "," << cfg.schedule << "," << cfg.chunk_size << ","
      << (cfg.master_compute ? 1 : 0) << "," << world_size << ","
      << cfg.width << "," << cfg.height << "," << total_ms << "," << compute_max << ","
      << compute_max << "," << (compute_sum / std::max<size_t>(1, metrics.size())) << ","
      << comm_sum << "," << io_sum << "," << yolo_sum << "," << idle_sum << ","
      << load_imbalance(metrics) << "," << avg_count << ",";
    if (correctness < 0) f << "";
    else f << correctness;
    f << "\n";
}

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

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.live) {
            MPI_Barrier(MPI_COMM_WORLD);
            run_live(cfg, rank, world_size);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 0;
        }
        auto tasks = make_tasks(cfg);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::steady_clock::now();
        std::string payload = cfg.schedule == "dynamic"
            ? run_dynamic(cfg, tasks, MPI_COMM_WORLD)
            : run_static(cfg, tasks, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0) {
            fs::create_directories(cfg.output);
            std::vector<Detection> detections;
            std::vector<Metrics> metrics;
            parse_payload(payload, detections, metrics);
            metrics = aggregate_metrics_by_rank(metrics);
            auto by_frame = nms_by_frame(cfg, detections);
            int correctness = -1;
            if (cfg.verify) {
                bool ok = verify_counts(cfg, tasks, by_frame, fs::path(cfg.output) / "correctness.txt");
                correctness = ok ? 1 : 0;
                std::cout << "CORRECTNESS_PASS=" << (ok ? "YES" : "NO") << "\n";
            }
            write_frame_counts(fs::path(cfg.output) / "frame_counts.csv", cfg, by_frame);
            write_bboxes(fs::path(cfg.output) / "bboxes.csv", by_frame);
            write_rank_metrics(fs::path(cfg.output) / "rank_metrics.csv", metrics);
            write_summary(fs::path(cfg.output) / "summary.csv", cfg, world_size, static_cast<int>(tasks.size()), total_ms, metrics, by_frame, correctness);
            std::cout << "YOLO_MPI_CPP_RUN_DONE=YES\n";
            std::cout << "RUN_DIR=" << cfg.output << "\n";
            std::cout << "SUMMARY_CSV=" << (fs::path(cfg.output) / "summary.csv").string() << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank=" << rank << " ERROR: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
