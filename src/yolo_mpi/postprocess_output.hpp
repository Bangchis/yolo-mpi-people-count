// Box merging, NMS, correctness verification, and CSV output.
// Rank 0 calls these after it has gathered detections from all ranks.

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
