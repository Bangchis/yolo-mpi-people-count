#pragma once

#include <mpi.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../core/metrics.hpp"
#include "../core/partition.hpp"
#include "../core/tensor.hpp"
#include "halo_exchange.hpp"
#include "local_conv.hpp"

namespace vgg11_mpi {

inline void broadcast_weights(ConvWeights &weights, int root, MPI_Comm comm) {
  int meta[3] = {weights.in_channels, weights.out_channels, weights.kernel};
  MPI_Bcast(meta, 3, MPI_INT, root, comm);

  if (weights.weight.empty()) {
    weights = ConvWeights(meta[0], meta[1], meta[2]);
  }

  MPI_Bcast(weights.weight.data(), static_cast<int>(weights.weight.size()), MPI_FLOAT, root, comm);
  MPI_Bcast(weights.bias.data(), static_cast<int>(weights.bias.size()), MPI_FLOAT, root, comm);
}

// One distributed 3x3 same-padding convolution layer.
//
// Rank 0 starts with the full feature map. It scatters 2D blocks to all ranks.
// Every rank exchanges a one-pixel halo with neighbors, computes its local
// convolution block, then sends the result back to rank 0 for the next layer.
inline Tensor distributed_conv3x3_same(const Tensor &root_input,
                                       ConvWeights weights,
                                       int grid_rows,
                                       int grid_cols,
                                       int layer_id,
                                       const std::string &halo_mode,
                                       MPI_Comm comm,
                                       LayerMetrics &layer_metrics,
                                       RankMetrics &rank_metrics) {
  int rank = 0;
  int world_size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &world_size);

  if (grid_rows * grid_cols != world_size) {
    throw std::runtime_error("grid_rows * grid_cols must equal MPI world size");
  }

  int shape[3] = {root_input.c, root_input.h, root_input.w};
  MPI_Bcast(shape, 3, MPI_INT, 0, comm);
  broadcast_weights(weights, 0, comm);

  const int channels = shape[0];
  const int height = shape[1];
  const int width = shape[2];
  const int grid_r = rank / grid_cols;
  const int grid_c = rank % grid_cols;
  const Block2D my_block = make_block(height, width, grid_rows, grid_cols, grid_r, grid_c);

  const double total_start = now_ms();
  const double scatter_start = now_ms();

  std::vector<float> my_core;
  int patch_meta[2] = {my_block.rows.size(), my_block.cols.size()};

  if (rank == 0) {
    for (int dst = 0; dst < world_size; ++dst) {
      const int dst_r = dst / grid_cols;
      const int dst_c = dst % grid_cols;
      const Block2D block = make_block(height, width, grid_rows, grid_cols, dst_r, dst_c);
      std::vector<float> core = pack_core_block(root_input, block);
      int meta[4] = {block.rows.size(), block.cols.size(), block.rows.start, block.cols.start};

      if (dst == 0) {
        my_core = std::move(core);
        patch_meta[0] = meta[0];
        patch_meta[1] = meta[1];
      } else {
        MPI_Send(meta, 4, MPI_INT, dst, 101, comm);
        MPI_Send(core.data(), static_cast<int>(core.size()), MPI_FLOAT, dst, 102, comm);
      }
    }
  } else {
    int meta[4] = {0, 0, 0, 0};
    MPI_Recv(meta, 4, MPI_INT, 0, 101, comm, MPI_STATUS_IGNORE);
    patch_meta[0] = meta[0];
    patch_meta[1] = meta[1];

    my_core.resize(static_cast<size_t>(channels) * patch_meta[0] * patch_meta[1]);
    MPI_Recv(my_core.data(), static_cast<int>(my_core.size()), MPI_FLOAT, 0, 102, comm, MPI_STATUS_IGNORE);
  }

  const double scatter_end = now_ms();

  Tensor patch_tensor = core_to_extended_patch(my_core, channels, patch_meta[0], patch_meta[1]);

  const double halo_start = now_ms();
  if (halo_mode == "blocking") {
    exchange_halo_blocking(patch_tensor, grid_rows, grid_cols, grid_r, grid_c, comm);
  } else if (halo_mode == "nonblocking") {
    exchange_halo_nonblocking(patch_tensor, grid_rows, grid_cols, grid_r, grid_c, comm);
  } else {
    throw std::runtime_error("unsupported halo mode: " + halo_mode);
  }
  const double halo_end = now_ms();

  const double compute_start = now_ms();
  Tensor local_output = local_conv_from_patch(patch_tensor, weights, patch_meta[0], patch_meta[1]);
  const double compute_end = now_ms();

  const double gather_start = now_ms();
  Tensor full_output;

  if (rank == 0) {
    full_output = Tensor(weights.out_channels, height, width);
    copy_local_output_into_full(full_output, local_output, my_block);

    for (int src = 1; src < world_size; ++src) {
      const int src_r = src / grid_cols;
      const int src_c = src % grid_cols;
      const Block2D block = make_block(height, width, grid_rows, grid_cols, src_r, src_c);
      const int count = weights.out_channels * block.rows.size() * block.cols.size();
      std::vector<float> buffer(static_cast<size_t>(count));

      MPI_Recv(buffer.data(), count, MPI_FLOAT, src, 103, comm, MPI_STATUS_IGNORE);
      Tensor src_output(weights.out_channels, block.rows.size(), block.cols.size());
      src_output.data = std::move(buffer);
      copy_local_output_into_full(full_output, src_output, block);
    }
  } else {
    MPI_Send(local_output.data.data(),
             static_cast<int>(local_output.data.size()),
             MPI_FLOAT,
             0,
             103,
             comm);
  }

  const double gather_end = now_ms();
  const double total_end = now_ms();

  rank_metrics.scatter_ms += scatter_end - scatter_start;
  rank_metrics.halo_ms += halo_end - halo_start;
  rank_metrics.compute_ms += compute_end - compute_start;
  rank_metrics.gather_ms += gather_end - gather_start;
  rank_metrics.total_ms += total_end - total_start;

  layer_metrics.layer_id = layer_id;
  layer_metrics.in_channels = channels;
  layer_metrics.out_channels = weights.out_channels;
  layer_metrics.height = height;
  layer_metrics.width = width;
  layer_metrics.scatter_ms = scatter_end - scatter_start;
  layer_metrics.halo_ms = halo_end - halo_start;
  layer_metrics.compute_ms = compute_end - compute_start;
  layer_metrics.gather_ms = gather_end - gather_start;
  layer_metrics.total_ms = total_end - total_start;

  return full_output;
}

}  // namespace vgg11_mpi
