#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vgg11_mpi/distributed_conv.hpp"
#include "vgg11_mpi/tensor.hpp"

namespace fs = std::filesystem;

namespace {

using vgg11_mpi::ConvWeights;
using vgg11_mpi::ErrorStats;
using vgg11_mpi::LayerMetrics;
using vgg11_mpi::RankMetrics;
using vgg11_mpi::Tensor;

struct ConvLayerSpec {
  int out_channels = 0;
  bool pool_after = false;
};

struct Config {
  int height = 64;
  int width = 64;
  std::string profile = "tiny";
  std::string output_dir = "results/vgg11_mpi";
  std::string grid = "auto";
  std::string halo_mode = "blocking";
  int check_serial = 1;
  int repeats = 1;
};

struct TopologyMetrics {
  int total_edges = 0;
  int intra_machine_edges = 0;
  int inter_machine_edges = 0;
  int cardinal_edges = 0;
  int diagonal_edges = 0;
};

void print_usage() {
  std::cerr
      << "Usage: vgg11_mpi [options]\n"
      << "  --height N              input feature-map height, default 64\n"
      << "  --width N               input feature-map width, default 64\n"
      << "  --profile tiny|small|full  VGG11 no-BN channel scale, default tiny\n"
      << "  --grid auto|RxC         MPI 2D process grid, default auto\n"
      << "  --halo-mode blocking|nonblocking  halo communication strategy, default blocking\n"
      << "  --output-dir DIR        output directory\n"
      << "  --check-serial 0|1      compare with serial stack on rank 0, default 1\n"
      << "  --repeats N             repeat distributed run and keep the last output\n";
}

Config parse_args(int argc, char **argv) {
  Config cfg;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--height") {
      cfg.height = std::stoi(require_value(arg));
    } else if (arg == "--width") {
      cfg.width = std::stoi(require_value(arg));
    } else if (arg == "--profile") {
      cfg.profile = require_value(arg);
    } else if (arg == "--grid") {
      cfg.grid = require_value(arg);
    } else if (arg == "--halo-mode") {
      cfg.halo_mode = require_value(arg);
    } else if (arg == "--output-dir") {
      cfg.output_dir = require_value(arg);
    } else if (arg == "--check-serial") {
      cfg.check_serial = std::stoi(require_value(arg));
    } else if (arg == "--repeats") {
      cfg.repeats = std::max(1, std::stoi(require_value(arg)));
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (cfg.height <= 0 || cfg.width <= 0) {
    throw std::runtime_error("height and width must be positive");
  }
  if (cfg.halo_mode != "blocking" && cfg.halo_mode != "nonblocking") {
    throw std::runtime_error("halo mode must be blocking or nonblocking");
  }

  return cfg;
}

std::vector<ConvLayerSpec> make_vgg11_no_bn_layers(const std::string &profile) {
  std::vector<int> channels;

  if (profile == "tiny") {
    channels = {8, 16, 32, 32, 64, 64, 64, 64};
  } else if (profile == "small") {
    channels = {16, 32, 64, 64, 128, 128, 128, 128};
  } else if (profile == "full") {
    // VGG11 without BatchNorm: 8 convolution layers.
    channels = {64, 128, 256, 256, 512, 512, 512, 512};
  } else {
    throw std::runtime_error("unsupported VGG11 profile: " + profile);
  }

  std::vector<ConvLayerSpec> layers;
  layers.reserve(channels.size());

  for (size_t i = 0; i < channels.size(); ++i) {
    // VGG11 pooling pattern: after conv 1, 2, 4, 6, and 8.
    const bool pool_after = (i == 0 || i == 1 || i == 3 || i == 5 || i == 7);
    layers.push_back({channels[i], pool_after});
  }

  return layers;
}

std::pair<int, int> choose_grid(const std::string &grid_arg, int world_size) {
  if (grid_arg != "auto") {
    const size_t x_pos = grid_arg.find('x');
    if (x_pos == std::string::npos) {
      throw std::runtime_error("grid must be auto or formatted like 3x4");
    }

    const int rows = std::stoi(grid_arg.substr(0, x_pos));
    const int cols = std::stoi(grid_arg.substr(x_pos + 1));
    if (rows <= 0 || cols <= 0 || rows * cols != world_size) {
      throw std::runtime_error("grid rows * cols must equal MPI world size");
    }
    return {rows, cols};
  }

  int dims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, dims);

  // Prefer rows <= cols for a natural left-to-right 2D mesh.
  int rows = std::min(dims[0], dims[1]);
  int cols = std::max(dims[0], dims[1]);
  return {rows, cols};
}

Tensor run_serial_stack(const Config &cfg, const std::vector<ConvLayerSpec> &layers) {
  Tensor current = vgg11_mpi::make_deterministic_input(3, cfg.height, cfg.width);
  int in_channels = 3;

  for (size_t i = 0; i < layers.size(); ++i) {
    ConvWeights weights =
        vgg11_mpi::make_deterministic_weights(static_cast<int>(i), in_channels, layers[i].out_channels);

    current = vgg11_mpi::serial_conv3x3_same(current, weights);
    vgg11_mpi::relu_in_place(current);

    if (layers[i].pool_after && current.h >= 2 && current.w >= 2) {
      current = vgg11_mpi::max_pool_2x2(current);
    }

    in_channels = layers[i].out_channels;
  }

  return current;
}

Tensor run_distributed_stack(const Config &cfg,
                             const std::vector<ConvLayerSpec> &layers,
                             int grid_rows,
                             int grid_cols,
                             std::vector<LayerMetrics> &layers_out,
                             RankMetrics &rank_metrics,
                             MPI_Comm comm) {
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  Tensor current;
  if (rank == 0) {
    current = vgg11_mpi::make_deterministic_input(3, cfg.height, cfg.width);
  }

  int in_channels = 3;
  layers_out.clear();

  for (size_t i = 0; i < layers.size(); ++i) {
    ConvWeights weights;
    if (rank == 0) {
      weights = vgg11_mpi::make_deterministic_weights(static_cast<int>(i), in_channels, layers[i].out_channels);
    }

    LayerMetrics local_layer;
    Tensor next = vgg11_mpi::distributed_conv3x3_same(current,
                                                      weights,
                                                      grid_rows,
                                                      grid_cols,
                                                      static_cast<int>(i),
                                                      cfg.halo_mode,
                                                      comm,
                                                      local_layer,
                                                      rank_metrics);

    double local_values[5] = {
        local_layer.scatter_ms,
        local_layer.halo_ms,
        local_layer.compute_ms,
        local_layer.gather_ms,
        local_layer.total_ms,
    };
    double max_values[5] = {0, 0, 0, 0, 0};
    MPI_Reduce(local_values, max_values, 5, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
      local_layer.scatter_ms = max_values[0];
      local_layer.halo_ms = max_values[1];
      local_layer.compute_ms = max_values[2];
      local_layer.gather_ms = max_values[3];
      local_layer.total_ms = max_values[4];
      layers_out.push_back(local_layer);

      vgg11_mpi::relu_in_place(next);
      if (layers[i].pool_after && next.h >= 2 && next.w >= 2) {
        next = vgg11_mpi::max_pool_2x2(next);
      }
      current = std::move(next);
    }

    in_channels = layers[i].out_channels;
  }

  return current;
}

std::vector<RankMetrics> gather_rank_metrics(const RankMetrics &local, MPI_Comm comm) {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  double local_values[5] = {
      local.scatter_ms,
      local.halo_ms,
      local.compute_ms,
      local.gather_ms,
      local.total_ms,
  };

  std::vector<double> all_values(static_cast<size_t>(world_size) * 5, 0.0);
  MPI_Gather(local_values, 5, MPI_DOUBLE, all_values.data(), 5, MPI_DOUBLE, 0, comm);

  char host[MPI_MAX_PROCESSOR_NAME] = {0};
  int host_len = 0;
  MPI_Get_processor_name(host, &host_len);

  std::vector<char> all_hosts(static_cast<size_t>(world_size) * MPI_MAX_PROCESSOR_NAME, '\0');
  MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hosts.data(), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, comm);

  std::vector<RankMetrics> rows;
  if (rank == 0) {
    rows.reserve(static_cast<size_t>(world_size));
    for (int r = 0; r < world_size; ++r) {
      RankMetrics row;
      row.rank = r;
      row.hostname = std::string(&all_hosts[static_cast<size_t>(r) * MPI_MAX_PROCESSOR_NAME]);
      row.scatter_ms = all_values[static_cast<size_t>(r) * 5 + 0];
      row.halo_ms = all_values[static_cast<size_t>(r) * 5 + 1];
      row.compute_ms = all_values[static_cast<size_t>(r) * 5 + 2];
      row.gather_ms = all_values[static_cast<size_t>(r) * 5 + 3];
      row.total_ms = all_values[static_cast<size_t>(r) * 5 + 4];
      rows.push_back(row);
    }
  }

  return rows;
}

void write_summary(const fs::path &path,
                   const Config &cfg,
                   int world_size,
                   int grid_rows,
                   int grid_cols,
                   int conv_layers,
                   double distributed_ms,
                   double serial_ms,
                   const ErrorStats &error) {
  std::ofstream out(path);
  out << "profile,height,width,world_size,grid_rows,grid_cols,conv_layers,"
      << "halo_mode,distributed_ms,serial_ms,max_abs_error,mean_abs_error,correct\n";
  out << cfg.profile << ','
      << cfg.height << ','
      << cfg.width << ','
      << world_size << ','
      << grid_rows << ','
      << grid_cols << ','
      << conv_layers << ','
      << cfg.halo_mode << ','
      << std::fixed << std::setprecision(6)
      << distributed_ms << ','
      << serial_ms << ','
      << error.max_abs << ','
      << error.mean_abs << ','
      << (error.max_abs < 1e-3 ? "YES" : "NO") << '\n';
}

void write_layers(const fs::path &path, const std::vector<LayerMetrics> &layers) {
  std::ofstream out(path);
  out << "layer_id,in_channels,out_channels,height,width,scatter_ms,halo_ms,compute_ms,gather_ms,total_ms\n";

  for (const LayerMetrics &row : layers) {
    out << row.layer_id << ','
        << row.in_channels << ','
        << row.out_channels << ','
        << row.height << ','
        << row.width << ','
        << std::fixed << std::setprecision(6)
        << row.scatter_ms << ','
        << row.halo_ms << ','
        << row.compute_ms << ','
        << row.gather_ms << ','
        << row.total_ms << '\n';
  }
}

void write_rank_metrics(const fs::path &path, const std::vector<RankMetrics> &rows) {
  double max_total = 0.0;
  for (const RankMetrics &row : rows) {
    max_total = std::max(max_total, row.total_ms);
  }

  std::ofstream out(path);
  out << "rank,hostname,scatter_ms,halo_ms,compute_ms,gather_ms,total_ms,idle_ms\n";

  for (const RankMetrics &row : rows) {
    const double idle_ms = std::max(0.0, max_total - row.total_ms);
    out << row.rank << ','
        << row.hostname << ','
        << std::fixed << std::setprecision(6)
        << row.scatter_ms << ','
        << row.halo_ms << ','
        << row.compute_ms << ','
        << row.gather_ms << ','
        << row.total_ms << ','
        << idle_ms << '\n';
  }
}

TopologyMetrics compute_topology_metrics(const std::vector<RankMetrics> &rows,
                                         int grid_rows,
                                         int grid_cols) {
  TopologyMetrics metrics;

  auto rank_at = [grid_cols](int row, int col) {
    return row * grid_cols + col;
  };

  const int directions[4][2] = {
      {0, 1},   // right
      {1, 0},   // down
      {1, 1},   // down-right
      {1, -1},  // down-left
  };

  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      const int a = rank_at(r, c);

      for (const auto &dir : directions) {
        const int nr = r + dir[0];
        const int nc = c + dir[1];
        if (nr < 0 || nr >= grid_rows || nc < 0 || nc >= grid_cols) {
          continue;
        }

        const int b = rank_at(nr, nc);
        const bool diagonal = (dir[0] != 0 && dir[1] != 0);
        const bool same_host = rows[static_cast<size_t>(a)].hostname == rows[static_cast<size_t>(b)].hostname;

        metrics.total_edges += 1;
        metrics.cardinal_edges += diagonal ? 0 : 1;
        metrics.diagonal_edges += diagonal ? 1 : 0;
        metrics.intra_machine_edges += same_host ? 1 : 0;
        metrics.inter_machine_edges += same_host ? 0 : 1;
      }
    }
  }

  return metrics;
}

void write_topology_mapping(const fs::path &path,
                            const std::vector<RankMetrics> &rows,
                            int grid_rows,
                            int grid_cols,
                            int height,
                            int width) {
  std::ofstream out(path);
  out << "rank,grid_row,grid_col,hostname,block_row_start,block_row_end,block_col_start,block_col_end\n";

  for (const RankMetrics &row : rows) {
    const int grid_r = row.rank / grid_cols;
    const int grid_c = row.rank % grid_cols;
    const auto block = vgg11_mpi::make_block(height, width, grid_rows, grid_cols, grid_r, grid_c);

    out << row.rank << ','
        << grid_r << ','
        << grid_c << ','
        << row.hostname << ','
        << block.rows.start << ','
        << block.rows.end << ','
        << block.cols.start << ','
        << block.cols.end << '\n';
  }
}

void write_topology_metrics(const fs::path &path, const TopologyMetrics &metrics) {
  const double inter_ratio =
      metrics.total_edges > 0 ? static_cast<double>(metrics.inter_machine_edges) / metrics.total_edges : 0.0;

  std::ofstream out(path);
  out << "total_edges,intra_machine_edges,inter_machine_edges,inter_machine_edge_ratio,"
      << "cardinal_edges,diagonal_edges\n";
  out << metrics.total_edges << ','
      << metrics.intra_machine_edges << ','
      << metrics.inter_machine_edges << ','
      << std::fixed << std::setprecision(6)
      << inter_ratio << ','
      << metrics.cardinal_edges << ','
      << metrics.diagonal_edges << '\n';
}

void write_topology_grid(const fs::path &path,
                         const std::vector<RankMetrics> &rows,
                         int grid_rows,
                         int grid_cols) {
  std::ofstream out(path);

  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      const int rank = r * grid_cols + c;
      std::string host = rows[static_cast<size_t>(rank)].hostname;
      const size_t dot = host.find('.');
      if (dot != std::string::npos) {
        host = host.substr(0, dot);
      }

      out << "P" << rank << "@" << host;
      if (c + 1 < grid_cols) {
        out << " | ";
      }
    }
    out << '\n';
  }
}

void write_manifest(const fs::path &path,
                    const Config &cfg,
                    int world_size,
                    int grid_rows,
                    int grid_cols,
                    const TopologyMetrics &topology) {
  std::ofstream out(path);
  out << "method=Method 2: VGG11 no-BN distributed convolution\n";
  out << "parallelism=data parallelism\n";
  out << "decomposition=2D feature-map block decomposition\n";
  out << "topology=2D Cartesian/mesh topology\n";
  out << "mapping=topology-aware row-major rank placement; use contiguous hostfile slots for physical locality\n";
  out << "communication=neighbor halo exchange plus root gather between layers\n";
  out << "halo_mode=" << cfg.halo_mode << '\n';
  out << "batch_norm=disabled\n";
  out << "profile=" << cfg.profile << '\n';
  out << "height=" << cfg.height << '\n';
  out << "width=" << cfg.width << '\n';
  out << "world_size=" << world_size << '\n';
  out << "grid_rows=" << grid_rows << '\n';
  out << "grid_cols=" << grid_cols << '\n';
  out << "halo_total_edges=" << topology.total_edges << '\n';
  out << "halo_intra_machine_edges=" << topology.intra_machine_edges << '\n';
  out << "halo_inter_machine_edges=" << topology.inter_machine_edges << '\n';
  out << "check_serial=" << cfg.check_serial << '\n';
  out << "repeats=" << cfg.repeats << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  try {
    const Config cfg = parse_args(argc, argv);
    const auto layers = make_vgg11_no_bn_layers(cfg.profile);
    const auto [grid_rows, grid_cols] = choose_grid(cfg.grid, world_size);

    RankMetrics rank_metrics;
    rank_metrics.rank = rank;
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    rank_metrics.hostname = std::string(host);

    std::vector<LayerMetrics> layer_metrics;
    Tensor distributed_output;

    MPI_Barrier(MPI_COMM_WORLD);
    const double distributed_start = vgg11_mpi::now_ms();
    for (int repeat = 0; repeat < cfg.repeats; ++repeat) {
      distributed_output = run_distributed_stack(cfg,
                                                 layers,
                                                 grid_rows,
                                                 grid_cols,
                                                 layer_metrics,
                                                 rank_metrics,
                                                 MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double distributed_end = vgg11_mpi::now_ms();
    const double distributed_ms = distributed_end - distributed_start;

    ErrorStats error;
    double serial_ms = 0.0;
    if (rank == 0 && cfg.check_serial != 0) {
      const double serial_start = vgg11_mpi::now_ms();
      Tensor serial_output = run_serial_stack(cfg, layers);
      const double serial_end = vgg11_mpi::now_ms();
      serial_ms = serial_end - serial_start;
      error = vgg11_mpi::compare_tensors(serial_output, distributed_output);
    }

    const std::vector<RankMetrics> all_rank_metrics = gather_rank_metrics(rank_metrics, MPI_COMM_WORLD);

    if (rank == 0) {
      const TopologyMetrics topology = compute_topology_metrics(all_rank_metrics, grid_rows, grid_cols);

      fs::create_directories(cfg.output_dir);
      write_summary(fs::path(cfg.output_dir) / "summary.csv",
                    cfg,
                    world_size,
                    grid_rows,
                    grid_cols,
                    static_cast<int>(layers.size()),
                    distributed_ms,
                    serial_ms,
                    error);
      write_layers(fs::path(cfg.output_dir) / "layer_metrics.csv", layer_metrics);
      write_rank_metrics(fs::path(cfg.output_dir) / "rank_metrics.csv", all_rank_metrics);
      write_topology_mapping(fs::path(cfg.output_dir) / "topology_mapping.csv",
                             all_rank_metrics,
                             grid_rows,
                             grid_cols,
                             cfg.height,
                             cfg.width);
      write_topology_metrics(fs::path(cfg.output_dir) / "topology_metrics.csv", topology);
      write_topology_grid(fs::path(cfg.output_dir) / "topology_grid.txt", all_rank_metrics, grid_rows, grid_cols);
      write_manifest(fs::path(cfg.output_dir) / "manifest.txt", cfg, world_size, grid_rows, grid_cols, topology);

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
