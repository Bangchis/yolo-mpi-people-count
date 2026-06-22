// Extra duplicate rule for close-up people crossing tile boundaries.
// It is disabled by default for crowded scenes because it can merge too much.
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

    // Tile-owner filter: a bbox belongs to the tile containing its center.
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
    // Highest-confidence boxes win; overlapping duplicate boxes are removed or merged.
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

