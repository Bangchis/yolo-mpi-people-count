#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vgg11_mpi {

struct Config {
  int height = 64;
  int width = 64;
  std::string profile = "tiny";
  std::string output_dir = "results/vgg11_mpi";
  std::string grid = "auto";
  std::string halo_mode = "blocking";
  std::string input_list;
  int input_limit = 0;
  int check_serial = 1;
  int repeats = 1;
};

inline void print_usage() {
  std::cerr
      << "Usage: vgg11_mpi [options]\n"
      << "  --height N              input feature-map height, default 64\n"
      << "  --width N               input feature-map width, default 64\n"
      << "  --profile tiny|small|full  VGG11 no-BN channel scale, default tiny\n"
      << "  --grid auto|RxC         MPI 2D process grid, default auto\n"
      << "  --halo-mode blocking|nonblocking  halo communication strategy, default blocking\n"
      << "  --input-list FILE       optional PPM image list for small image-dataset runs\n"
      << "  --input-limit N         use only first N images from --input-list, default all\n"
      << "  --output-dir DIR        output directory\n"
      << "  --check-serial 0|1      compare with serial stack on rank 0, default 1\n"
      << "  --repeats N             repeat distributed run and keep the last output\n";
}

inline Config parse_args(int argc, char **argv) {
  Config cfg;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };

    if (arg == "--height") {
      cfg.height = std::stoi(require_value(arg));
    } else if (arg == "--width") {
      cfg.width = std::stoi(require_value(arg));
    } else if (arg == "--profile") {
      cfg.profile = require_value(arg);
    } else if (arg == "--grid") {
      cfg.grid = require_value(arg);
    } else if (arg == "--halo-mode") {
      cfg.halo_mode = require_value(arg);
    } else if (arg == "--input-list") {
      cfg.input_list = require_value(arg);
    } else if (arg == "--input-limit") {
      cfg.input_limit = std::max(0, std::stoi(require_value(arg)));
    } else if (arg == "--output-dir") {
      cfg.output_dir = require_value(arg);
    } else if (arg == "--check-serial") {
      cfg.check_serial = std::stoi(require_value(arg));
    } else if (arg == "--repeats") {
      cfg.repeats = std::max(1, std::stoi(require_value(arg)));
    } else if (arg == "--help" || arg == "-h") {
      print_usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (cfg.height <= 0 || cfg.width <= 0) {
    throw std::runtime_error("height and width must be positive");
  }
  if (cfg.halo_mode != "blocking" && cfg.halo_mode != "nonblocking") {
    throw std::runtime_error("halo mode must be blocking or nonblocking");
  }

  return cfg;
}

inline std::pair<int, int> choose_grid(const std::string &grid_arg, int world_size) {
  if (grid_arg != "auto") {
    const size_t x_pos = grid_arg.find('x');
    if (x_pos == std::string::npos) {
      throw std::runtime_error("grid must be auto or formatted like 3x4");
    }

    const int rows = std::stoi(grid_arg.substr(0, x_pos));
    const int cols = std::stoi(grid_arg.substr(x_pos + 1));
    if (rows <= 0 || cols <= 0 || rows * cols != world_size) {
      throw std::runtime_error("grid rows * cols must equal MPI world size");
    }
    return {rows, cols};
  }

  int dims[2] = {0, 0};
  MPI_Dims_create(world_size, 2, dims);

  // Prefer rows <= cols so the mesh is easy to draw in the report.
  int rows = std::min(dims[0], dims[1]);
  int cols = std::max(dims[0], dims[1]);
  return {rows, cols};
}

}  // namespace vgg11_mpi
