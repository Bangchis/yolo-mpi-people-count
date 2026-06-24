#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

#include "../core/config.hpp"
#include "../core/metrics.hpp"
#include "../core/partition.hpp"
#include "../core/tensor.hpp"

namespace vgg11_mpi {

namespace fs = std::filesystem;

inline void write_summary(const fs::path &path,
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

inline void write_layers(const fs::path &path, const std::vector<LayerMetrics> &layers) {
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

inline void write_rank_metrics(const fs::path &path, const std::vector<RankMetrics> &rows) {
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

inline void write_topology_mapping(const fs::path &path,
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
    const auto block = make_block(height, width, grid_rows, grid_cols, grid_r, grid_c);

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

inline void write_topology_metrics(const fs::path &path, const TopologyMetrics &metrics) {
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

inline void write_topology_grid(const fs::path &path,
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

inline void write_manifest(const fs::path &path,
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

}  // namespace vgg11_mpi
