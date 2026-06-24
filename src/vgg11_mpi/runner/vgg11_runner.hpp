#pragma once

#include <mpi.h>

#include <utility>
#include <vector>

#include "../conv/distributed_conv.hpp"
#include "../core/config.hpp"
#include "../core/metrics.hpp"
#include "../core/tensor.hpp"
#include "../core/vgg11_layers.hpp"

namespace vgg11_mpi {

inline Tensor run_serial_stack(const Config &cfg, const std::vector<ConvLayerSpec> &layers) {
  Tensor current = make_deterministic_input(3, cfg.height, cfg.width);
  int in_channels = 3;

  for (size_t i = 0; i < layers.size(); ++i) {
    ConvWeights weights = make_deterministic_weights(static_cast<int>(i), in_channels, layers[i].out_channels);

    current = serial_conv3x3_same(current, weights);
    relu_in_place(current);

    if (layers[i].pool_after && current.h >= 2 && current.w >= 2) {
      current = max_pool_2x2(current);
    }

    in_channels = layers[i].out_channels;
  }

  return current;
}

inline Tensor run_distributed_stack(const Config &cfg,
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
    current = make_deterministic_input(3, cfg.height, cfg.width);
  }

  int in_channels = 3;
  layers_out.clear();

  for (size_t i = 0; i < layers.size(); ++i) {
    ConvWeights weights;
    if (rank == 0) {
      weights = make_deterministic_weights(static_cast<int>(i), in_channels, layers[i].out_channels);
    }

    LayerMetrics local_layer;
    Tensor next = distributed_conv3x3_same(current,
                                           weights,
                                           grid_rows,
                                           grid_cols,
                                           static_cast<int>(i),
                                           cfg.halo_mode,
                                           comm,
                                           local_layer,
                                           rank_metrics);

    // Rank 0 records the maximum layer time across ranks, which is the real layer wall time.
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

      relu_in_place(next);
      if (layers[i].pool_after && next.h >= 2 && next.w >= 2) {
        next = max_pool_2x2(next);
      }
      current = std::move(next);
    }

    in_channels = layers[i].out_channels;
  }

  return current;
}

inline std::vector<RankMetrics> gather_rank_metrics(const RankMetrics &local, MPI_Comm comm) {
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

}  // namespace vgg11_mpi
