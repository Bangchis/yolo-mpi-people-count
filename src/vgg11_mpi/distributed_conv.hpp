#pragma once

#include <mpi.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "partition.hpp"
#include "tensor.hpp"

namespace vgg11_mpi {

struct LayerMetrics {
  int layer_id = 0;
  int in_channels = 0;
  int out_channels = 0;
  int height = 0;
  int width = 0;
  double scatter_ms = 0.0;
  double halo_ms = 0.0;
  double compute_ms = 0.0;
  double gather_ms = 0.0;
  double total_ms = 0.0;
};

struct RankMetrics {
  int rank = 0;
  std::string hostname;
  double scatter_ms = 0.0;
  double halo_ms = 0.0;
  double compute_ms = 0.0;
  double gather_ms = 0.0;
  double total_ms = 0.0;
};

inline double now_ms() {
  return MPI_Wtime() * 1000.0;
}

inline void broadcast_weights(ConvWeights &weights, int root, MPI_Comm comm) {
  int meta[3] = {weights.in_channels, weights.out_channels, weights.kernel};
  MPI_Bcast(meta, 3, MPI_INT, root, comm);

  if (weights.weight.empty()) {
    weights = ConvWeights(meta[0], meta[1], meta[2]);
  }

  MPI_Bcast(weights.weight.data(), static_cast<int>(weights.weight.size()), MPI_FLOAT, root, comm);
  MPI_Bcast(weights.bias.data(), static_cast<int>(weights.bias.size()), MPI_FLOAT, root, comm);
}

inline std::vector<float> pack_core_block(const Tensor &input, const Block2D &block) {
  std::vector<float> core(static_cast<size_t>(input.c) * block.rows.size() * block.cols.size(), 0.0f);

  for (int c = 0; c < input.c; ++c) {
    for (int y = 0; y < block.rows.size(); ++y) {
      for (int x = 0; x < block.cols.size(); ++x) {
        const size_t dst = (static_cast<size_t>(c) * block.rows.size() + y) * block.cols.size() + x;
        core[dst] = input.at(c, block.rows.start + y, block.cols.start + x);
      }
    }
  }

  return core;
}

inline Tensor core_to_extended_patch(const std::vector<float> &core, int channels, int core_h, int core_w) {
  Tensor tensor(channels, core_h + 2, core_w + 2);

  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < core_h; ++y) {
      for (int x = 0; x < core_w; ++x) {
        const size_t src = (static_cast<size_t>(c) * core_h + y) * core_w + x;
        tensor.at(c, y + 1, x + 1) = core[src];
      }
    }
  }

  return tensor;
}

inline Tensor local_conv_from_patch(const Tensor &patch, const ConvWeights &weights, int core_h, int core_w) {
  Tensor output(weights.out_channels, core_h, core_w);

  for (int oc = 0; oc < weights.out_channels; ++oc) {
    for (int y = 0; y < core_h; ++y) {
      for (int x = 0; x < core_w; ++x) {
        float sum = weights.bias[static_cast<size_t>(oc)];

        for (int ic = 0; ic < weights.in_channels; ++ic) {
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              // The patch already contains the one-pixel halo, so the core point (y,x)
              // corresponds to patch coordinate (y+1,x+1).
              sum += patch.at(ic, y + ky, x + kx) * weights.w_at(oc, ic, ky, kx);
            }
          }
        }

        output.at(oc, y, x) = sum;
      }
    }
  }

  return output;
}

inline int cart_neighbor(int grid_rows, int grid_cols, int row, int col, int drow, int dcol) {
  const int nr = row + drow;
  const int nc = col + dcol;

  if (nr < 0 || nr >= grid_rows || nc < 0 || nc >= grid_cols) {
    return MPI_PROC_NULL;
  }

  return nr * grid_cols + nc;
}

inline std::vector<float> pack_row(const Tensor &patch, int y) {
  std::vector<float> row(static_cast<size_t>(patch.c) * (patch.w - 2), 0.0f);

  for (int c = 0; c < patch.c; ++c) {
    for (int x = 1; x < patch.w - 1; ++x) {
      row[static_cast<size_t>(c) * (patch.w - 2) + (x - 1)] = patch.at(c, y, x);
    }
  }

  return row;
}

inline void unpack_row(Tensor &patch, int y, const std::vector<float> &row) {
  for (int c = 0; c < patch.c; ++c) {
    for (int x = 1; x < patch.w - 1; ++x) {
      patch.at(c, y, x) = row[static_cast<size_t>(c) * (patch.w - 2) + (x - 1)];
    }
  }
}

inline std::vector<float> pack_col(const Tensor &patch, int x) {
  std::vector<float> col(static_cast<size_t>(patch.c) * (patch.h - 2), 0.0f);

  for (int c = 0; c < patch.c; ++c) {
    for (int y = 1; y < patch.h - 1; ++y) {
      col[static_cast<size_t>(c) * (patch.h - 2) + (y - 1)] = patch.at(c, y, x);
    }
  }

  return col;
}

inline void unpack_col(Tensor &patch, int x, const std::vector<float> &col) {
  for (int c = 0; c < patch.c; ++c) {
    for (int y = 1; y < patch.h - 1; ++y) {
      patch.at(c, y, x) = col[static_cast<size_t>(c) * (patch.h - 2) + (y - 1)];
    }
  }
}

inline std::vector<float> pack_corner(const Tensor &patch, int y, int x) {
  std::vector<float> corner(static_cast<size_t>(patch.c), 0.0f);

  for (int c = 0; c < patch.c; ++c) {
    corner[static_cast<size_t>(c)] = patch.at(c, y, x);
  }

  return corner;
}

inline void unpack_corner(Tensor &patch, int y, int x, const std::vector<float> &corner) {
  for (int c = 0; c < patch.c; ++c) {
    patch.at(c, y, x) = corner[static_cast<size_t>(c)];
  }
}

inline void sendrecv_vector(std::vector<float> &send_buf,
                            int dst,
                            int send_tag,
                            std::vector<float> &recv_buf,
                            int src,
                            int recv_tag,
                            MPI_Comm comm) {
  MPI_Sendrecv(send_buf.data(),
               static_cast<int>(send_buf.size()),
               MPI_FLOAT,
               dst,
               send_tag,
               recv_buf.data(),
               static_cast<int>(recv_buf.size()),
               MPI_FLOAT,
               src,
               recv_tag,
               comm,
               MPI_STATUS_IGNORE);
}

inline void exchange_halo_blocking(Tensor &patch,
                                   int grid_rows,
                                   int grid_cols,
                                   int grid_r,
                                   int grid_c,
                                   MPI_Comm comm) {
  const int top = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, 0);
  const int bottom = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, 0);
  const int left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 0, -1);
  const int right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 0, 1);

  std::vector<float> send_top = pack_row(patch, 1);
  std::vector<float> recv_top(send_top.size(), 0.0f);
  std::vector<float> send_bottom = pack_row(patch, patch.h - 2);
  std::vector<float> recv_bottom(send_bottom.size(), 0.0f);

  sendrecv_vector(send_top, top, 200, recv_top, top, 201, comm);
  sendrecv_vector(send_bottom, bottom, 201, recv_bottom, bottom, 200, comm);
  unpack_row(patch, 0, recv_top);
  unpack_row(patch, patch.h - 1, recv_bottom);

  std::vector<float> send_left = pack_col(patch, 1);
  std::vector<float> recv_left(send_left.size(), 0.0f);
  std::vector<float> send_right = pack_col(patch, patch.w - 2);
  std::vector<float> recv_right(send_right.size(), 0.0f);

  sendrecv_vector(send_left, left, 210, recv_left, left, 211, comm);
  sendrecv_vector(send_right, right, 211, recv_right, right, 210, comm);
  unpack_col(patch, 0, recv_left);
  unpack_col(patch, patch.w - 1, recv_right);

  const int top_left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, -1);
  const int top_right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, 1);
  const int bottom_left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, -1);
  const int bottom_right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, 1);

  std::vector<float> send_tl = pack_corner(patch, 1, 1);
  std::vector<float> recv_tl(send_tl.size(), 0.0f);
  std::vector<float> send_br = pack_corner(patch, patch.h - 2, patch.w - 2);
  std::vector<float> recv_br(send_br.size(), 0.0f);
  sendrecv_vector(send_tl, top_left, 220, recv_tl, top_left, 221, comm);
  sendrecv_vector(send_br, bottom_right, 221, recv_br, bottom_right, 220, comm);
  unpack_corner(patch, 0, 0, recv_tl);
  unpack_corner(patch, patch.h - 1, patch.w - 1, recv_br);

  std::vector<float> send_tr = pack_corner(patch, 1, patch.w - 2);
  std::vector<float> recv_tr(send_tr.size(), 0.0f);
  std::vector<float> send_bl = pack_corner(patch, patch.h - 2, 1);
  std::vector<float> recv_bl(send_bl.size(), 0.0f);
  sendrecv_vector(send_tr, top_right, 222, recv_tr, top_right, 223, comm);
  sendrecv_vector(send_bl, bottom_left, 223, recv_bl, bottom_left, 222, comm);
  unpack_corner(patch, 0, patch.w - 1, recv_tr);
  unpack_corner(patch, patch.h - 1, 0, recv_bl);
}

inline void exchange_halo_nonblocking(Tensor &patch,
                                      int grid_rows,
                                      int grid_cols,
                                      int grid_r,
                                      int grid_c,
                                      MPI_Comm comm) {
  const int top = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, 0);
  const int bottom = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, 0);
  const int left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 0, -1);
  const int right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 0, 1);
  const int top_left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, -1);
  const int top_right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, -1, 1);
  const int bottom_left = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, -1);
  const int bottom_right = cart_neighbor(grid_rows, grid_cols, grid_r, grid_c, 1, 1);

  std::vector<float> send_top = pack_row(patch, 1);
  std::vector<float> recv_top(send_top.size(), 0.0f);
  std::vector<float> send_bottom = pack_row(patch, patch.h - 2);
  std::vector<float> recv_bottom(send_bottom.size(), 0.0f);
  std::vector<float> send_left = pack_col(patch, 1);
  std::vector<float> recv_left(send_left.size(), 0.0f);
  std::vector<float> send_right = pack_col(patch, patch.w - 2);
  std::vector<float> recv_right(send_right.size(), 0.0f);
  std::vector<float> send_tl = pack_corner(patch, 1, 1);
  std::vector<float> recv_tl(send_tl.size(), 0.0f);
  std::vector<float> send_tr = pack_corner(patch, 1, patch.w - 2);
  std::vector<float> recv_tr(send_tr.size(), 0.0f);
  std::vector<float> send_bl = pack_corner(patch, patch.h - 2, 1);
  std::vector<float> recv_bl(send_bl.size(), 0.0f);
  std::vector<float> send_br = pack_corner(patch, patch.h - 2, patch.w - 2);
  std::vector<float> recv_br(send_br.size(), 0.0f);

  std::vector<MPI_Request> requests;
  requests.reserve(16);

  auto post_recv = [&](std::vector<float> &buffer, int src, int tag) {
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Irecv(buffer.data(), static_cast<int>(buffer.size()), MPI_FLOAT, src, tag, comm, &request);
    requests.push_back(request);
  };

  auto post_send = [&](std::vector<float> &buffer, int dst, int tag) {
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Isend(buffer.data(), static_cast<int>(buffer.size()), MPI_FLOAT, dst, tag, comm, &request);
    requests.push_back(request);
  };

  // These tags match exchange_halo_blocking, but all transfers are posted at once.
  post_recv(recv_top, top, 201);
  post_recv(recv_bottom, bottom, 200);
  post_recv(recv_left, left, 211);
  post_recv(recv_right, right, 210);
  post_recv(recv_tl, top_left, 221);
  post_recv(recv_tr, top_right, 223);
  post_recv(recv_bl, bottom_left, 222);
  post_recv(recv_br, bottom_right, 220);

  post_send(send_top, top, 200);
  post_send(send_bottom, bottom, 201);
  post_send(send_left, left, 210);
  post_send(send_right, right, 211);
  post_send(send_tl, top_left, 220);
  post_send(send_tr, top_right, 222);
  post_send(send_bl, bottom_left, 223);
  post_send(send_br, bottom_right, 221);

  MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

  unpack_row(patch, 0, recv_top);
  unpack_row(patch, patch.h - 1, recv_bottom);
  unpack_col(patch, 0, recv_left);
  unpack_col(patch, patch.w - 1, recv_right);
  unpack_corner(patch, 0, 0, recv_tl);
  unpack_corner(patch, 0, patch.w - 1, recv_tr);
  unpack_corner(patch, patch.h - 1, 0, recv_bl);
  unpack_corner(patch, patch.h - 1, patch.w - 1, recv_br);
}

inline void copy_local_output_into_full(Tensor &full, const Tensor &local, const Block2D &block) {
  for (int c = 0; c < local.c; ++c) {
    for (int y = 0; y < local.h; ++y) {
      for (int x = 0; x < local.w; ++x) {
        full.at(c, block.rows.start + y, block.cols.start + x) = local.at(c, y, x);
      }
    }
  }
}

// Distributed Conv2D used by Method 2.
//
// Rank 0 owns the full input tensor. For each convolution layer it scatters one
// core 2D feature-map block to every rank. Neighboring ranks then exchange
// halo rows/columns/corners in the 2D mesh, compute local convolution, and rank
// 0 gathers output blocks for the next VGG11 layer.
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
