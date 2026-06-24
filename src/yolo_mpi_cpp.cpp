#include "yolo_mpi/core/types.hpp"

// Course-readable order: core config -> detectors -> MPI -> postprocess -> output -> live.
#include "yolo_mpi/core/system.hpp"
#include "yolo_mpi/core/config.hpp"
#include "yolo_mpi/core/tasks.hpp"
#include "yolo_mpi/detector/mock_command.hpp"
#include "yolo_mpi/detector/yolo_worker_process.hpp"
#include "yolo_mpi/detector/runner.hpp"
#include "yolo_mpi/detector/payload.hpp"
#include "yolo_mpi/mpi/communication.hpp"
#include "yolo_mpi/mpi/static_scheduler.hpp"
#include "yolo_mpi/mpi/protocol.hpp"
#include "yolo_mpi/postprocess/geometry.hpp"
#include "yolo_mpi/postprocess/duplicate_rules.hpp"
#include "yolo_mpi/postprocess/temporal.hpp"
#include "yolo_mpi/postprocess/frame_merge.hpp"
#include "yolo_mpi/output/csv.hpp"
#include "yolo_mpi/live/io.hpp"
#include "yolo_mpi/live/frame_processing.hpp"
#include "yolo_mpi/live/runner.hpp"

// Program entrypoint for both offline video benchmark mode and live camera mode.
int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);

    // Every process becomes one MPI rank from this point.
    MPI_Init(&argc, &argv);
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    try {
        // All ranks parse the same command line so they agree on workload settings.
        Config cfg = parse_args(argc, argv);
        if (cfg.live) {
            // Live mode: rank 0 captures camera frames, other ranks receive JPEG tiles.
            MPI_Barrier(MPI_COMM_WORLD);
            run_live(cfg, rank, world_size);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 0;
        }
        // Offline mode: build all frame/tile tasks before scheduling.
        auto tasks = make_tasks(cfg);

        // Start timing after all ranks have built the same task list.
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::steady_clock::now();

        // Static scheduler returns serialized DET/MET rows on rank 0.
        std::string payload = run_static(cfg, tasks, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::steady_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (rank == 0) {
            // Only rank 0 owns final merge, correctness check, and CSV output.
            fs::create_directories(cfg.output);
            std::vector<Detection> detections;
            std::vector<Metrics> metrics;
            parse_payload(payload, detections, metrics);
            metrics = aggregate_metrics_by_rank(metrics);

            // Global postprocess happens once after rank 0 has all boxes.
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
            write_summary(
                fs::path(cfg.output) / "summary.csv",
                cfg,
                world_size,
                static_cast<int>(tasks.size()),
                total_ms,
                metrics,
                by_frame,
                correctness
            );
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
