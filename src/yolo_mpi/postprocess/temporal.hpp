// Decide whether a current live box likely matches a box from the previous frame.
static bool temporal_match(const Config& cfg, const Detection& current, const Detection& previous) {
    if (current.cls != previous.cls) {
        return false;
    }

    if (iou(current, previous) >= 0.15) {
        return true;
    }

    if (intersection_over_smaller(current, previous) >= cfg.live_temporal_ios) {
        return true;
    }

    // If IoU is weak, compare normalized center distance and axis overlap.
    double cw = box_width(current), pw = box_width(previous);
    double ch = box_height(current), ph = box_height(previous);
    double max_w = std::max(cw, pw);
    double max_h = std::max(ch, ph);

    if (max_w <= 0 || max_h <= 0) {
        return false;
    }

    double cx = (current.x1 + current.x2) * 0.5;
    double cy = (current.y1 + current.y2) * 0.5;
    double px = (previous.x1 + previous.x2) * 0.5;
    double py = (previous.y1 + previous.y2) * 0.5;
    double ndx = std::abs(cx - px) / max_w;
    double ndy = std::abs(cy - py) / max_h;
    double ox = axis_overlap_ratio(current.x1, current.x2, previous.x1, previous.x2);
    double oy = axis_overlap_ratio(current.y1, current.y2, previous.y1, previous.y2);

    // Centers must be close and at least one axis should overlap.
    return ndx <= cfg.live_temporal_center &&
           ndy <= cfg.live_temporal_center &&
           (ox > 0.10 || oy > 0.10);
}

// Merge current-frame duplicate boxes that appear to come from one previous object.
static std::vector<Detection> temporal_dedup_against_previous(
    const Config& cfg,
    const std::vector<Detection>& current,
    const std::vector<Detection>& previous,
    int frame_width,
    int frame_height
) {
    if (!cfg.live_temporal_dedup || current.size() < 2 || previous.empty()) {
        return current;
    }

    // Live-only guard: if one person flickers into multiple boxes across frames,
    // merge current boxes that match the same previous-frame box.
    std::vector<bool> used(current.size(), false);
    std::vector<Detection> out;
    out.reserve(current.size());

    for (const auto& prev : previous) {
        std::vector<size_t> matches;

        // Find all current boxes that could correspond to this previous object.
        for (size_t i = 0; i < current.size(); ++i) {
            if (!used[i] && temporal_match(cfg, current[i], prev)) {
                matches.push_back(i);
            }
        }

        if (matches.size() < 2) {
            continue;
        }

        Detection merged = current[matches.front()];
        used[matches.front()] = true;

        // Multiple current boxes matched one previous object, so merge them.
        for (size_t j = 1; j < matches.size(); ++j) {
            merged = merge_duplicate_detection(merged, current[matches[j]]);
            used[matches[j]] = true;
        }
        out.push_back(merged);
    }

    for (size_t i = 0; i < current.size(); ++i) {
        // Preserve boxes that did not belong to any temporal merge group.
        if (!used[i]) {
            out.push_back(current[i]);
        }
    }

    return nms(cfg, out, frame_width, frame_height);
}
