#pragma once

#include <string>
#include <vector>

#include "../core/metrics.hpp"

namespace vgg11_mpi {

inline TopologyMetrics compute_topology_metrics(const std::vector<RankMetrics> &rows,
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

}  // namespace vgg11_mpi
