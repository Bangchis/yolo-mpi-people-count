#pragma once

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

// All command-line options are stored here after parse_args().
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

// Offline task: one frame tile in original-frame coordinates.
struct Task {
    int task_id = 0;
    int frame_id = 0;
    int tile_id = 0;
    int x1 = 0;
    int y1 = 0;
    int x2 = 0;
    int y2 = 0;
};

// Detection boxes are already remapped to original-frame coordinates.
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

// One or more rows per rank before aggregation; rank 0 later groups them.
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

// Live task: same tile metadata plus a base64 JPEG tile.
struct ImageTask {
    Task task;
    std::string encoded_jpeg;
};

// One live frame from the master camera, including all tile tasks.
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
