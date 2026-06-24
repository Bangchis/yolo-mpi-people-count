#pragma once

#include <mpi.h>

#include <string>

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

struct TopologyMetrics {
  int total_edges = 0;
  int intra_machine_edges = 0;
  int inter_machine_edges = 0;
  int cardinal_edges = 0;
  int diagonal_edges = 0;
};

struct ImageMetrics {
  int image_index = 0;
  std::string image_path;
  int height = 0;
  int width = 0;
  double distributed_ms = 0.0;
  double serial_ms = 0.0;
  double max_abs_error = 0.0;
  double mean_abs_error = 0.0;
  std::string correct = "NO";
};

inline double now_ms() {
  return MPI_Wtime() * 1000.0;
}

}  // namespace vgg11_mpi
