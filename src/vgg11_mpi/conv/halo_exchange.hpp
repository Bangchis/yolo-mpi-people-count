#pragma once

#include <mpi.h>

#include <vector>

#include "../core/tensor.hpp"

namespace vgg11_mpi {

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

  // Rows and columns are enough for 4-neighbor convolution dependencies.
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

  // Corners are needed because a 3x3 kernel reads diagonal neighbors too.
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

  // Receives are posted first, then sends. This avoids unnecessary blocking.
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

}  // namespace vgg11_mpi
