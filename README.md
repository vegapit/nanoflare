# Nanoflare

Nanoflare is a header-only C++17 library designed to be a fast and lightweight alternative to Libtorch for real-time inference of Pytorch models. It was originally developed for audio processing but can be extended at will.

## Usage

There are currently 2 ways of using Nanoflare

### Using built-in neural network architectures

The Nanoflare workflow that covers using models provided with the library consists in:
* Calibrate the model in *Python* using the `pynanoflare` modules
* Serialise to *JSON* using the `generate_doc` method
* Build its corresponding *C++* object by loading the *JSON* document into the `Nanoflare::ModelBuilder` class

### Using custom architectures

If you would like to build your own neural network architecture, you would just:
* Write the *Python* module using the `pynanoflare` module
* Add a `generate_doc` function that handles *JSON* serialisation
* Write a *C++* equivalent version that derives from the `Nanoflare::BaseModel` virtual abstract class.

Examining the code of built-in models and `Nanoflare::ModelBuilder` should get you started very quickly.

Models can be registered in the `Nanoflare::ModelBuilder` at runtime, so new models can be defined in their own *Python* and *C++* modules while still being managed by this class.

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
* FiLM
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

The easiest way to use the library is to add it as a Git submodule to your project,add it as a sub-directory, and register the include folders with the `NANOFLARE_INCLUDE_DIRS` variable:

```
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanoflare)

include_directories(${NANOFLARE_INCLUDE_DIRS})
```

To get the inference to run at optimal speed, do not forget to set optimisation tags to the compiler e.g. -march=native in OSX

## Tests and Benchmark

The tests are handled by the [Catch2](https://github.com/catchorg/Catch2.git) testing framework also defined as a Git submodule and use Libtorch as reference.

When configuring the tests, pass the path to the Libtorch directory to CMake as a `LIBTORCH_DIR` variable and define the `NANOFLARE_TESTING` variable:

```
cmake -B build -DNANOFLARE_TESTING=ON -DLIBTORCH_DIR=<path/to/libtorch>
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

The `pynanoflare` folder contains the Python modules implementing the Pytorch modules that can be replicated with this library. The `generate_test_data.py` scripts generates a new set of data used in the accuracy testing.