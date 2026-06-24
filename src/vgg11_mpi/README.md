# Method 2 Source Layout

Method 2 is the VGG11 no-BatchNorm distributed convolution experiment. It is
separate from Method 1 YOLO task parallelism.

```text
src/vgg11_mpi/
  core/
    config.hpp         command-line options and MPI grid selection
    image_dataset.hpp  PPM image-list reader for small image-dataset runs
    metrics.hpp        layer/rank/topology timing structs
    partition.hpp      2D block decomposition helpers
    tensor.hpp         simple C x H x W tensor, serial conv, ReLU, pooling
    vgg11_layers.hpp   VGG11 no-BN layer profiles

  conv/
    local_conv.hpp       local block packing and local 3x3 convolution
    halo_exchange.hpp    blocking and nonblocking halo exchange
    distributed_conv.hpp scatter -> halo -> local conv -> gather

  runner/
    vgg11_runner.hpp   serial stack, distributed stack, rank metric gather

  output/
    csv.hpp            summary/layer/rank/topology CSV writers
    topology.hpp       halo-neighbor topology analysis
```

`src/vgg11_mpi.cpp` is intentionally small. It only initializes MPI, runs the
experiment, checks correctness against the serial stack, and writes outputs.

The old top-level headers `tensor.hpp`, `partition.hpp`, and
`distributed_conv.hpp` are kept as short compatibility includes.

For real-image smoke tests, the binary supports:

```bash
build/vgg11_mpi --height 64 --width 64 --input-list data/vgg11-tiny-images/image_list.txt
```

Only rank 0 reads the image files. Other ranks receive their feature-map blocks
through MPI, exactly like the synthetic benchmark path.
