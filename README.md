# Nanoflare

Nanoflare is a header-only C++17 library designed to be a fast and lightweight alternative to Libtorch for real-time inference of Pytorch models. It was originally developed for audio processing but can be extended at will.

## Usage & Dependencies

The library uses [Eigen3](https://gitlab.com/libeigen/eigen.git) for fast matrix computation, and [nlohmann::json](https://github.com/nlohmann/json.git) for saving and loading models to file. Both are defined as Git submodules and are consequently built with the library.

## Tests and Benchmark

The tests are handled by the [Catch2](https://github.com/catchorg/Catch2.git) testing framework also defined as a Git submodule and use Libtorch as reference.

When configuring the tests, pass the path to the Libtorch directory to CMake as a `LIBTORCH_DIR` variable:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLIBTORCH_DIR=<path/to/libtorch>
```

Launch the build and run accuracy tests:

```
cmake --build build
cd build
make test
```

The benchmarks comparing the processing speed of the library with Libtorch is run through:

```
./tests/models_benchmarking
```

Libtorch can be quite slow for the first few runs post-load so to make it fairer, a few preliminary warm-up runs are performed before measuring its performance. When that is done, Libtorch inference runtime is close to Nanoflare's on the testing machine.

## Python  modules

The `pytorch` folder contains the Python modules implementing the Pytorch models that can be replicated with Nanoflare. The `generate_test_data.py` scripts generates a new set of data used in the accuracy testing.