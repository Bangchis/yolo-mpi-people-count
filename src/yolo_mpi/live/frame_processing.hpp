// Single-machine live path, mostly used for debugging without node1/node2.
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
    // Rank 0 sends JPEG tiles to workers and gathers bbox payloads back.
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
    // Optional full-frame anchor helps close-up camera cases where tiling splits one person.
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
