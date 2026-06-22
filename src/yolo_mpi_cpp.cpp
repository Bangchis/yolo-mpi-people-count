#include "yolo_mpi/common.hpp"

// The implementation is split by responsibility so students can read it in
// course-report order: configuration -> detector -> MPI -> merge/output -> live.
#include "yolo_mpi/config_and_tasks.hpp"
#include "yolo_mpi/detector_worker.hpp"
#include "yolo_mpi/mpi_scheduling.hpp"
#include "yolo_mpi/postprocess_output.hpp"
#include "yolo_mpi/live_pipeline.hpp"

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        Config cfg = parse_args(argc, argv);
        if (cfg.live) {
            MPI_Barrier(MPI_COMM_WORLD);
            run_live(cfg, rank, world_size);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 0;
        }
        auto tasks = make_tasks(cfg);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::steady_clock::now();
        std::string payload = cfg.schedule == "dynamic"
            ? run_dynamic(cfg, tasks, MPI_COMM_WORLD)
            : run_static(cfg, tasks, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0) {
            fs::create_directories(cfg.output);
            std::vector<Detection> detections;
            std::vector<Metrics> metrics;
            parse_payload(payload, detections, metrics);
            metrics = aggregate_metrics_by_rank(metrics);
            auto by_frame = nms_by_frame(cfg, detections);
            int correctness = -1;
            if (cfg.verify) {
                bool ok = verify_counts(cfg, tasks, by_frame, fs::path(cfg.output) / "correctness.txt");
                correctness = ok ? 1 : 0;
                std::cout << "CORRECTNESS_PASS=" << (ok ? "YES" : "NO") << "\n";
            }
            write_frame_counts(fs::path(cfg.output) / "frame_counts.csv", cfg, by_frame);
            write_bboxes(fs::path(cfg.output) / "bboxes.csv", by_frame);
            write_rank_metrics(fs::path(cfg.output) / "rank_metrics.csv", metrics);
            write_summary(fs::path(cfg.output) / "summary.csv", cfg, world_size, static_cast<int>(tasks.size()), total_ms, metrics, by_frame, correctness);
            std::cout << "YOLO_MPI_CPP_RUN_DONE=YES\n";
            std::cout << "RUN_DIR=" << cfg.output << "\n";
            std::cout << "SUMMARY_CSV=" << (fs::path(cfg.output) / "summary.csv").string() << "\n";
        }
    } catch (const std::exception& exc) {
        std::cerr << "rank=" << rank << " ERROR: " << exc.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
