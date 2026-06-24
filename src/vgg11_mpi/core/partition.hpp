#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace vgg11_mpi {

struct Range {
  int start = 0;
  int end = 0;

  int size() const {
    return end - start;
  }
};

struct Block2D {
  Range rows;
  Range cols;
};

inline Range split_1d(int total, int parts, int index) {
  if (parts <= 0 || index < 0 || index >= parts) {
    throw std::runtime_error("invalid split_1d arguments");
  }

  const int base = total / parts;
  const int extra = total % parts;

  // Give the first "extra" ranks one more element so all blocks differ by at most 1.
  const int start = index * base + std::min(index, extra);
  const int size = base + (index < extra ? 1 : 0);

  return {start, start + size};
}

inline Block2D make_block(int height, int width, int grid_rows, int grid_cols, int grid_r, int grid_c) {
  return {
      split_1d(height, grid_rows, grid_r),
      split_1d(width, grid_cols, grid_c),
  };
}

inline std::vector<Block2D> all_blocks(int height, int width, int grid_rows, int grid_cols) {
  std::vector<Block2D> blocks;
  blocks.reserve(static_cast<size_t>(grid_rows * grid_cols));

  for (int r = 0; r < grid_rows; ++r) {
    for (int c = 0; c < grid_cols; ++c) {
      blocks.push_back(make_block(height, width, grid_rows, grid_cols, r, c));
    }
  }

  return blocks;
}

}  // namespace vgg11_mpi
