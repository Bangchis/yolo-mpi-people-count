// Dynamic scheduling: rank 0 owns a task queue and gives new work to whichever
// worker just finished. This is the report's main load-balancing strategy.

// Run dynamic MPI mode: rank 0 schedules work, workers return one payload per task.
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

        // Prime the queue with one task per worker. If rank 0 also computes,
        // keep at least one task available locally for the master.
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

        // Receive one worker result and either issue another task or stop that worker.
        auto receive_worker_result = [&](bool block, std::ostringstream& all) -> bool {
            if (active <= 0) return false;
            if (!block) {
                // Non-blocking poll lets rank 0 compute while workers are busy.
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
                // Rank 0 is not just a coordinator; it can run YOLO too.
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
