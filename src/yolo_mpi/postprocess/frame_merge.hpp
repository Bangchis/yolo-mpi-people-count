// Apply all per-frame postprocessing: tile ownership first, then global NMS.
static std::vector<Detection> merge_frame_detections(
    const Config& cfg,
    const std::vector<Detection>& detections,
    int frame_width,
    int frame_height
) {
    auto filtered = apply_tile_owner_filter(cfg, detections, frame_width, frame_height);
    return nms(cfg, filtered, frame_width, frame_height);
}

// Group detections by frame and run frame-level postprocess for each group.
static std::map<int, std::vector<Detection>> nms_by_frame(const Config& cfg, const std::vector<Detection>& detections) {
    std::map<int, std::vector<Detection>> grouped;
    for (const auto& det : detections) grouped[det.frame_id].push_back(det);
    for (auto& [frame, dets] : grouped) {
        (void)frame;
        dets = merge_frame_detections(cfg, dets, cfg.width, cfg.height);
    }
    return grouped;
}
