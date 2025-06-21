# Nanoflare

Nanoflare is a header-only C++17 library designed to be a fast and lightweight alternative to Libtorch for real-time inference of Pytorch models. It was originally developed for audio processing but can be extended at will.

## Usage

At this time, Pytorch models can not be directly imported. Instead, the workflow consists in defining a neural network architecture in Python using the Pytorch wrapper class provided, calibrate it, serialise it as JSON and load it from file as a C++ object using the `ModelBuilder` factory function.

## Layers & Models

The basic layer types currently available are:

* BatchNorm1d
* Conv1d
* GRU
* GRUCell
* Linear
* LSTM
* LSTMCell
* PReLU

They were used to define the following custom block types:

* CausalDilatedConv1d
* ConvClipper
* MicroTCNBlock
* PlainSequential
* ResidualBlock
* TCNBlock

Which were in turn used to define the following models:

* ConvWaveshaper
* MicroTCN
* ResRNN e.g. ResGRU or ResLSTM
* TCN
* WaveNet

## Dependencies

The library uses [Eigen3](https://gitlab.com/libeigen/eigen.git) for fast matrix computation, and [nlohmann::json](https://github.com/nlohmann/json.git) for saving and loading models to file. Both are defined as Git submodules and built with the library.

The easiest way to use the library is to add it as a Git submodule to your project and register its `include` directory. For example in CMake:

```
include_directories(${CMAKE_SOURCE_DIR}/nanoflare/nanoflare/include)
```

To get the inference to run at optimal speed, do not forget to set optimisation tags to the compiler e.g. -march=native in OSX

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