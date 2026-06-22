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
