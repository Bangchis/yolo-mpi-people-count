#pragma once

#include <vector>

#include "../core/partition.hpp"
#include "../core/tensor.hpp"

namespace vgg11_mpi {

inline std::vector<float> pack_core_block(const Tensor &input, const Block2D &block) {
  std::vector<float> core(static_cast<size_t>(input.c) * block.rows.size() * block.cols.size(), 0.0f);

  // Copy only the owned feature-map block. Halo pixels are filled later by MPI.
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
  Tensor patch(channels, core_h + 2, core_w + 2);

  // Put the local block in the middle. The outer border is the one-pixel halo.
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < core_h; ++y) {
      for (int x = 0; x < core_w; ++x) {
        const size_t src = (static_cast<size_t>(c) * core_h + y) * core_w + x;
        patch.at(c, y + 1, x + 1) = core[src];
      }
    }
  }

  return patch;
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
              // The patch already includes halo, so no boundary branch is needed here.
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

inline void copy_local_output_into_full(Tensor &full, const Tensor &local, const Block2D &block) {
  for (int c = 0; c < local.c; ++c) {
    for (int y = 0; y < local.h; ++y) {
      for (int x = 0; x < local.w; ++x) {
        full.at(c, block.rows.start + y, block.cols.start + x) = local.at(c, y, x);
      }
    }
  }
}

}  // namespace vgg11_mpi
