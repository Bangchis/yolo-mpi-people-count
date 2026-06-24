#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace vgg11_mpi {

struct ConvLayerSpec {
  int out_channels = 0;
  bool pool_after = false;
};

inline std::vector<ConvLayerSpec> make_vgg11_no_bn_layers(const std::string &profile) {
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

}  // namespace vgg11_mpi
