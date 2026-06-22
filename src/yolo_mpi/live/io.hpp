// Live camera pipeline.
// Rank 0 captures frames, distributes JPEG tiles, gathers boxes, and displays.

// Build argv for the Python camera/tile source process.
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

// Build argv for the Python live viewer process.
static std::vector<std::string> viewer_args(const Config& cfg) {
    return {
        cfg.python_bin,
        cfg.viewer_script,
        "--display", cfg.live_view ? "1" : "0",
        "--save-dir", (fs::path(cfg.output) / "live_frames").string(),
    };
}

// Parse a FRAME line emitted by camera_tile_source.py.
static bool parse_camera_frame_line(const std::string& line, CameraFrame& frame) {
    std::istringstream iss(line);
    std::string tag;
    if (!(iss >> tag >> frame.frame_id >> frame.width >> frame.height >> frame.capture_ms >> frame.encoded_frame)) {
        return false;
    }
    return tag == "FRAME";
}

// Parse one TILE line emitted after a FRAME line.
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

// Read one complete frame plus all of its JPEG tile tasks from the camera process.
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

// Serialize one live JPEG tile so rank 0 can send it to a worker rank.
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

// Parse a serialized live JPEG tile received by a worker rank.
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

// Send a full frame and its final boxes to the viewer process.
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

// Write per-frame live latency/count records.
static void write_live_events(const fs::path& path, const std::vector<LiveFrameEvent>& events) {
    std::ofstream f(path);
    f << "frame_id,person_count,tasks,capture_ms,latency_ms\n";
    f << std::fixed << std::setprecision(4);
    for (const auto& event : events) {
        f << event.frame_id << "," << event.person_count << "," << event.tasks << ","
          << event.capture_ms << "," << event.latency_ms << "\n";
    }
}
