# Nanoflare

Nanoflare is a header-only C++17 extensible library designed to be a fast and lightweight alternative to Libtorch for real-time inference of Pytorch models. It was originally developed for using calibrated Pytorch models in audio plugins.

## Usage

Nanoflare consists of a Python library located in the `pynanoflare` folder which acts as a wrapper for various Pytorch modules, and a corresponding header-only C++ library located in the `include` folder.

**Calibrating and exporting a Nanoflare compatible Pytorch model to C++** is easy and involves the following steps:

* Calibrating the model implemented in *Python* using the `pynanoflare` module
* Serialising it to *JSON* with the `generate_doc` method
* Loading its *C++* instance using the `Nanoflare::ModelBuilder` class

**If you would like to use your own neural network architecture,** you would just:
* Define its *Python* class using the `pynanoflare` module
* Add a `generate_doc` function that handles its *JSON* serialisation
* Write a *C++* equivalent version that derives from the `Nanoflare::BaseModel` virtual abstract class.

New models can be trained and exported as any other built-in network architectures. Examining the Python and C++ code for the models provided and `Nanoflare::ModelBuilder` are great ressources for understanding how the code is structured.

Models are registered to `Nanoflare::ModelBuilder` at runtime, so new models can be defined in their own *Python* and *C++* modules while still being managed by this class.

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
* FiLM
* MicroTCNBlock
* PlainSequential
* ResidualBlock
* TCNBlock

Which were in turn used to define the following models:

* HammersteinWiener
* MicroTCN
* ResRNN e.g. ResGRU or ResLSTM
* TCN
* WaveNet

## Building and Dependencies

The library uses [Eigen3](https://gitlab.com/libeigen/eigen.git) for fast matrix computation, and [nlohmann::json](https://github.com/nlohmann/json.git) for saving and loading models to file. Both are defined as Git submodules and built with the library.

A simple way of using the library is to register it as a Git submodule to your project, add it as a sub-directory, and define the include folders with the `NANOFLARE_INCLUDE_DIRS` variable:

```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanoflare)

include_directories(${NANOFLARE_INCLUDE_DIRS})
```

## Tests and Benchmark

The tests are handled by the [Catch2](https://github.com/catchorg/Catch2.git) testing framework also defined as a Git submodule and use Libtorch as reference.

When configuring the tests, pass the path to the Libtorch directory to CMake as a `LIBTORCH_DIR` variable and define the `NANOFLARE_TESTING` variable:

```shell
mkdir build && cd build
cmake .. -DNANOFLARE_TESTING=ON -DLIBTORCH_DIR=<path/to/libtorch>
```

Launch the build and run accuracy tests:

```shell
cmake --build .
make test
```

The benchmarks comparing the processing speed of the library with Libtorch is run through:

```shell
./tests/models_benchmarking
```

Libtorch can be quite slow for the first few runs post-load so to make it fairer, a few preliminary warm-up runs are performed before measuring its performance.

> On the testing machine, Nanoflare is about **25-30% faster** than Libtorch/TorchscriptJIT across all neural network architectures available.

The `generate_test_data.py` scripts generates a new set of data used in the accuracy testing.