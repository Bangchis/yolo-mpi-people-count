#include <mpi.h>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vgg11_mpi/core/config.hpp"
#include "vgg11_mpi/core/metrics.hpp"
#include "vgg11_mpi/core/tensor.hpp"
#include "vgg11_mpi/core/vgg11_layers.hpp"
#include "vgg11_mpi/output/csv.hpp"
#include "vgg11_mpi/output/topology.hpp"
#include "vgg11_mpi/runner/vgg11_runner.hpp"

namespace fs = std::filesystem;

namespace {

vgg11_mpi::RankMetrics make_local_rank_metrics(int rank) {
  vgg11_mpi::RankMetrics metrics;
  metrics.rank = rank;

  char host[MPI_MAX_PROCESSOR_NAME] = {0};
  int host_len = 0;
  MPI_Get_processor_name(host, &host_len);
  metrics.hostname = std::string(host);

  return metrics;
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  try {
    const vgg11_mpi::Config cfg = vgg11_mpi::parse_args(argc, argv);
    const auto layers = vgg11_mpi::make_vgg11_no_bn_layers(cfg.profile);
    const auto [grid_rows, grid_cols] = vgg11_mpi::choose_grid(cfg.grid, world_size);

    vgg11_mpi::RankMetrics rank_metrics = make_local_rank_metrics(rank);
    std::vector<vgg11_mpi::LayerMetrics> layer_metrics;
    vgg11_mpi::Tensor distributed_output;

    // Run the distributed VGG11 no-BN stack. Rank 0 gathers each layer output.
    MPI_Barrier(MPI_COMM_WORLD);
    const double distributed_start = vgg11_mpi::now_ms();
    for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
      distributed_output = vgg11_mpi::run_distributed_stack(cfg,
                                                            layers,
                                                            grid_rows,
                                                            grid_cols,
                                                            layer_metrics,
                                                            rank_metrics,
                                                            MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double distributed_ms = vgg11_mpi::now_ms() - distributed_start;

    // Rank 0 checks correctness against the serial VGG11 no-BN implementation.
    vgg11_mpi::ErrorStats error;
    double serial_ms = 0.0;
    if (rank == 0 && cfg.check_serial != 0) {
      const double serial_start = vgg11_mpi::now_ms();
      const vgg11_mpi::Tensor serial_output = vgg11_mpi::run_serial_stack(cfg, layers);
      serial_ms = vgg11_mpi::now_ms() - serial_start;
      error = vgg11_mpi::compare_tensors(serial_output, distributed_output);
    }

    const std::vector<vgg11_mpi::RankMetrics> all_rank_metrics =
        vgg11_mpi::gather_rank_metrics(rank_metrics, MPI_COMM_WORLD);

    if (rank == 0) {
      const vgg11_mpi::TopologyMetrics topology =
          vgg11_mpi::compute_topology_metrics(all_rank_metrics, grid_rows, grid_cols);

      fs::create_directories(cfg.output_dir);
      vgg11_mpi::write_summary(fs::path(cfg.output_dir) / "summary.csv",
                               cfg,
                               world_size,
                               grid_rows,
                               grid_cols,
                               static_cast<int>(layers.size()),
                               distributed_ms,
                               serial_ms,
                               error);
      vgg11_mpi::write_layers(fs::path(cfg.output_dir) / "layer_metrics.csv", layer_metrics);
      vgg11_mpi::write_rank_metrics(fs::path(cfg.output_dir) / "rank_metrics.csv", all_rank_metrics);
      vgg11_mpi::write_topology_mapping(fs::path(cfg.output_dir) / "topology_mapping.csv",
                                        all_rank_metrics,
                                        grid_rows,
                                        grid_cols,
                                        cfg.height,
                                        cfg.width);
      vgg11_mpi::write_topology_metrics(fs::path(cfg.output_dir) / "topology_metrics.csv", topology);
      vgg11_mpi::write_topology_grid(fs::path(cfg.output_dir) / "topology_grid.txt",
                                     all_rank_metrics,
                                     grid_rows,
                                     grid_cols);
      vgg11_mpi::write_manifest(fs::path(cfg.output_dir) / "manifest.txt",
                                cfg,
                                world_size,
                                grid_rows,
                                grid_cols,
                                topology);

      std::cout << "VGG11_MPI_DONE=YES\n";
      std::cout << "VGG11_MPI_DIR=" << cfg.output_dir << "\n";
      std::cout << "VGG11_MPI_CORRECT=" << (error.max_abs < 1e-3 ? "YES" : "NO") << "\n";
      std::cout << "VGG11_MPI_MAX_ABS_ERROR=" << std::fixed << std::setprecision(8) << error.max_abs << "\n";
      std::cout << "VGG11_MPI_INTER_MACHINE_HALO_EDGES=" << topology.inter_machine_edges << "\n";
    }
  } catch (const std::exception &ex) {
    std::cerr << "VGG11_MPI_ERROR rank=" << rank << " message=" << ex.what() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
