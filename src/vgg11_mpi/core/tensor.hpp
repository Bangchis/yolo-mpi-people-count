#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vgg11_mpi {

// Small channel-first tensor for Method 2: C x H x W, contiguous in memory.
struct Tensor {
  int c = 0;
  int h = 0;
  int w = 0;
  std::vector<float> data;

  Tensor() = default;

  Tensor(int channels, int height, int width)
      : c(channels),
        h(height),
        w(width),
        data(static_cast<size_t>(channels) * height * width, 0.0f) {}

  size_t index(int channel, int y, int x) const {
    return (static_cast<size_t>(channel) * h + y) * w + x;
  }

  float &at(int channel, int y, int x) {
    return data[index(channel, y, x)];
  }

  float at(int channel, int y, int x) const {
    return data[index(channel, y, x)];
  }

  size_t size() const {
    return data.size();
  }
};

struct ConvWeights {
  int in_channels = 0;
  int out_channels = 0;
  int kernel = 3;
  std::vector<float> weight;
  std::vector<float> bias;

  ConvWeights() = default;

  ConvWeights(int cin, int cout, int k)
      : in_channels(cin),
        out_channels(cout),
        kernel(k),
        weight(static_cast<size_t>(cout) * cin * k * k, 0.0f),
        bias(static_cast<size_t>(cout), 0.0f) {}

  size_t weight_index(int out_c, int in_c, int ky, int kx) const {
    return (((static_cast<size_t>(out_c) * in_channels + in_c) * kernel + ky) * kernel + kx);
  }

  float w_at(int out_c, int in_c, int ky, int kx) const {
    return weight[weight_index(out_c, in_c, ky, kx)];
  }
};

inline Tensor make_deterministic_input(int channels, int height, int width) {
  Tensor input(channels, height, width);

  // Deterministic values make serial vs distributed correctness repeatable.
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const float v = std::sin(0.013f * static_cast<float>((c + 1) * 17 + y * 3 + x));
        input.at(c, y, x) = v;
      }
    }
  }

  return input;
}

inline ConvWeights make_deterministic_weights(int layer_id, int cin, int cout, int kernel = 3) {
  ConvWeights weights(cin, cout, kernel);

  // Small weights avoid exploding values after several VGG-style layers.
  for (size_t i = 0; i < weights.weight.size(); ++i) {
    const float v = std::sin(0.007f * static_cast<float>(i + 101 * (layer_id + 1)));
    weights.weight[i] = 0.035f * v;
  }

  for (int c = 0; c < cout; ++c) {
    weights.bias[static_cast<size_t>(c)] = 0.01f * std::cos(0.11f * static_cast<float>(layer_id + c));
  }

  return weights;
}

inline void relu_in_place(Tensor &tensor) {
  for (float &value : tensor.data) {
    value = std::max(0.0f, value);
  }
}

inline Tensor max_pool_2x2(const Tensor &input) {
  const int out_h = input.h / 2;
  const int out_w = input.w / 2;
  Tensor output(input.c, out_h, out_w);

  for (int c = 0; c < input.c; ++c) {
    for (int y = 0; y < out_h; ++y) {
      for (int x = 0; x < out_w; ++x) {
        float best = input.at(c, 2 * y, 2 * x);
        best = std::max(best, input.at(c, 2 * y + 1, 2 * x));
        best = std::max(best, input.at(c, 2 * y, 2 * x + 1));
        best = std::max(best, input.at(c, 2 * y + 1, 2 * x + 1));
        output.at(c, y, x) = best;
      }
    }
  }

  return output;
}

inline Tensor serial_conv3x3_same(const Tensor &input, const ConvWeights &weights) {
  if (weights.kernel != 3) {
    throw std::runtime_error("serial_conv3x3_same only supports kernel=3");
  }
  if (input.c != weights.in_channels) {
    throw std::runtime_error("input channels do not match convolution weights");
  }

  Tensor output(weights.out_channels, input.h, input.w);

  for (int oc = 0; oc < weights.out_channels; ++oc) {
    for (int y = 0; y < input.h; ++y) {
      for (int x = 0; x < input.w; ++x) {
        float sum = weights.bias[static_cast<size_t>(oc)];

        for (int ic = 0; ic < weights.in_channels; ++ic) {
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int yy = y + ky - 1;
              const int xx = x + kx - 1;

              // Same padding: samples outside the feature map are zero.
              if (yy < 0 || yy >= input.h || xx < 0 || xx >= input.w) {
                continue;
              }

              sum += input.at(ic, yy, xx) * weights.w_at(oc, ic, ky, kx);
            }
          }
        }

        output.at(oc, y, x) = sum;
      }
    }
  }

  return output;
}

struct ErrorStats {
  double max_abs = 0.0;
  double mean_abs = 0.0;
};

inline ErrorStats compare_tensors(const Tensor &expected, const Tensor &actual) {
  if (expected.c != actual.c || expected.h != actual.h || expected.w != actual.w) {
    throw std::runtime_error("cannot compare tensors with different shapes");
  }

  ErrorStats stats;
  double sum = 0.0;

  for (size_t i = 0; i < expected.data.size(); ++i) {
    const double err = std::abs(static_cast<double>(expected.data[i]) -
                                static_cast<double>(actual.data[i]));
    stats.max_abs = std::max(stats.max_abs, err);
    sum += err;
  }

  if (!expected.data.empty()) {
    stats.mean_abs = sum / static_cast<double>(expected.data.size());
  }

  return stats;
}

}  // namespace vgg11_mpi
