// MPI scheduling and communication.
// Static mode uses block-cyclic mapping; dynamic mode uses a master-worker queue.

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
