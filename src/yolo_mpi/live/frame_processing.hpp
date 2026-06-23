// Single-machine live path, mostly used for debugging without node1/node2.

// Process all live tiles on rank 0 only.
static std::vector<Detection> process_live_frame_locally(
    const Config& cfg,
    DetectorRunner& detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics
) {
    std::vector<Detection> frame_detections;

    for (const auto& image_task : frame.tiles) {
        // Local debug mode still uses the same payload parser as MPI mode.
        auto payload = process_one_image_task_payload(cfg, detector, image_task, 0);
        parse_payload(payload, frame_detections, metrics);
    }

    return merge_frame_detections(cfg, frame_detections, frame.width, frame.height);
}

// Distribute one live frame's tile tasks from rank 0 to worker ranks and merge results.
static std::vector<Detection> process_live_frame_distributed(
    const Config& cfg,
    DetectorRunner* local_detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics,
    int world_size
) {
    // Rank 0 sends JPEG tiles to workers and gathers bbox payloads back.
    const int task_tag = 50;
    const int result_tag = 70;
    const int total = static_cast<int>(frame.tiles.size());
    int next = 0;
    int completed = 0;
    int active = 0;
    std::vector<Detection> frame_detections;

    // If local_detector is present, rank 0 also processes live tiles.
    const bool master_compute = local_detector != nullptr;

    int initial_worker_tasks = std::min(world_size - 1, total);

    if (master_compute) {
        initial_worker_tasks = std::min(world_size - 1, std::max(0, total - 1));
    }

    for (int worker = 1; worker < world_size && next < initial_worker_tasks; ++worker) {
        // Send one tile to each worker at startup.
        send_string(serialize_image_task(frame.tiles[next++]), worker, task_tag);
        active += 1;
    }

    while (completed < total) {
        if (master_compute && next < total) {
            // Rank 0 can process a tile while worker ranks are busy.
            auto payload = process_one_image_task_payload(cfg, *local_detector, frame.tiles[next++], 0);
            parse_payload(payload, frame_detections, metrics);

            completed += 1;
        }

        if (active > 0) {
            MPI_Status status;
            // Wait for whichever worker finishes first.
            auto payload = recv_string(MPI_ANY_SOURCE, result_tag, &status);
            parse_payload(payload, frame_detections, metrics);
            completed += 1;

            int worker = status.MPI_SOURCE;
            active -= 1;

            bool reserve_one_for_master = master_compute && (total - next) <= 1;

            if (next < total && !reserve_one_for_master) {
                // Refill that same worker with the next unscheduled tile.
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

// Run YOLO once on the full frame as an anchor for close-up camera cases.
static std::vector<Detection> process_live_frame_anchor(
    const Config& cfg,
    DetectorRunner& detector,
    const CameraFrame& frame,
    std::vector<Metrics>& metrics
) {
    // Optional full-frame anchor helps close-up camera cases where tiling splits one person.
    ImageTask full_frame;
    full_frame.task.task_id = frame.frame_id;
    full_frame.task.frame_id = frame.frame_id;
    full_frame.task.tile_id = -1;
    full_frame.task.x1 = 0;
    full_frame.task.y1 = 0;
    full_frame.task.x2 = frame.width;
    full_frame.task.y2 = frame.height;
    // Use the full frame JPEG, not a cropped tile JPEG.
    full_frame.encoded_jpeg = frame.encoded_frame;

    std::vector<Detection> detections;
    auto payload = process_one_image_task_payload(cfg, detector, full_frame, 0);
    parse_payload(payload, detections, metrics);

    return merge_frame_detections(cfg, detections, frame.width, frame.height);
}

// Prefer full-frame anchor boxes, then add tile boxes not already covered by anchors.
static std::vector<Detection> merge_anchor_and_tile_detections(
    const Config& cfg,
    const std::vector<Detection>& anchors,
    const std::vector<Detection>& tiles,
    int frame_width,
    int frame_height
) {
    if (cfg.live_anchor_policy == "anchor-only") {
        // Most stable live display: trust only the full-frame YOLO result.
        return merge_frame_detections(cfg, anchors, frame_width, frame_height);
    }

    if (cfg.live_anchor_policy == "anchor-gate" && !anchors.empty()) {
        // Crowd-safe mode: tile ranks still compute, but the full-frame pass
        // decides the visible count. This prevents tile false positives from
        // turning three people into many duplicated people.
        return merge_frame_detections(cfg, anchors, frame_width, frame_height);
    }

    if (cfg.live_anchor_policy == "anchor-gate" && anchors.empty()) {
        // If the full-frame pass misses everyone, fall back to tile results so
        // the display does not get stuck at zero.
        return merge_frame_detections(cfg, tiles, frame_width, frame_height);
    }

    std::vector<Detection> combined = anchors;

    for (const auto& tile_det : tiles) {
        bool covered_by_anchor = false;

        for (const auto& anchor : anchors) {
            // If the full-frame detection already covers this tile box, skip it.
            if (duplicate_detection(cfg, tile_det, anchor, frame_width, frame_height)) {
                covered_by_anchor = true;
                break;
            }
        }

        if (!covered_by_anchor) {
            combined.push_back(tile_det);
        }
    }

    return merge_frame_detections(cfg, combined, frame_width, frame_height);
}
