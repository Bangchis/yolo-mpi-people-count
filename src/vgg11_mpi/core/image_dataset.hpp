#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.hpp"

namespace vgg11_mpi {

namespace fs = std::filesystem;

inline std::string trim_copy(const std::string &text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }

  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }

  return text.substr(begin, end - begin);
}

inline std::vector<fs::path> read_image_list(const fs::path &list_path, int limit) {
  std::ifstream in(list_path);
  if (!in) {
    throw std::runtime_error("cannot open image list: " + list_path.string());
  }

  std::vector<fs::path> paths;
  const fs::path base_dir = list_path.parent_path();

  std::string line;
  while (std::getline(in, line)) {
    line = trim_copy(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }

    fs::path path(line);
    if (path.is_relative()) {
      path = base_dir / path;
    }

    paths.push_back(path);
    if (limit > 0 && static_cast<int>(paths.size()) >= limit) {
      break;
    }
  }

  if (paths.empty()) {
    throw std::runtime_error("image list is empty: " + list_path.string());
  }

  return paths;
}

inline std::string read_ppm_token(std::istream &in) {
  std::string token;
  char ch = '\0';

  while (in.get(ch)) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      continue;
    }
    if (ch == '#') {
      in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    token.push_back(ch);
    break;
  }

  while (in.get(ch)) {
    if (std::isspace(static_cast<unsigned char>(ch))) {
      break;
    }
    token.push_back(ch);
  }

  if (token.empty()) {
    throw std::runtime_error("unexpected end of PPM header");
  }

  return token;
}

inline Tensor read_ppm_rgb_tensor(const fs::path &path, int expected_height, int expected_width) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open PPM image: " + path.string());
  }

  const std::string magic = read_ppm_token(in);
  const int width = std::stoi(read_ppm_token(in));
  const int height = std::stoi(read_ppm_token(in));
  const int max_value = std::stoi(read_ppm_token(in));

  if (magic != "P6" && magic != "P3") {
    throw std::runtime_error("only P6/P3 PPM images are supported: " + path.string());
  }
  if (max_value != 255) {
    throw std::runtime_error("only 8-bit PPM images are supported: " + path.string());
  }
  if ((expected_height > 0 && height != expected_height) ||
      (expected_width > 0 && width != expected_width)) {
    throw std::runtime_error("image size does not match --height/--width: " + path.string());
  }

  Tensor image(3, height, width);

  if (magic == "P6") {
    std::vector<unsigned char> bytes(static_cast<size_t>(height) * width * 3);
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size())) {
      throw std::runtime_error("truncated PPM image: " + path.string());
    }

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const size_t idx = (static_cast<size_t>(y) * width + x) * 3;
        image.at(0, y, x) = static_cast<float>(bytes[idx + 0]) / 255.0f;
        image.at(1, y, x) = static_cast<float>(bytes[idx + 1]) / 255.0f;
        image.at(2, y, x) = static_cast<float>(bytes[idx + 2]) / 255.0f;
      }
    }
  } else {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        image.at(0, y, x) = static_cast<float>(std::stoi(read_ppm_token(in))) / 255.0f;
        image.at(1, y, x) = static_cast<float>(std::stoi(read_ppm_token(in))) / 255.0f;
        image.at(2, y, x) = static_cast<float>(std::stoi(read_ppm_token(in))) / 255.0f;
      }
    }
  }

  return image;
}

}  // namespace vgg11_mpi
