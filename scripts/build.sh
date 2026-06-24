#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p build
if [[ -n "${MPICXX:-}" ]]; then
  compiler="$MPICXX"
elif [[ -x /opt/homebrew/bin/mpicxx ]]; then
  compiler=/opt/homebrew/bin/mpicxx
else
  compiler=mpicxx
fi

"$compiler" -std=c++17 -O2 -Wall -Wextra -pedantic \
  src/yolo_mpi_cpp.cpp \
  -o build/yolo_mpi_cpp

"$compiler" -std=c++17 -O2 -Wall -Wextra -pedantic \
  src/vgg11_mpi.cpp \
  -o build/vgg11_mpi

echo "BUILD_DONE=YES"
echo "BINARY=build/yolo_mpi_cpp"
echo "METHOD2_BINARY=build/vgg11_mpi"
