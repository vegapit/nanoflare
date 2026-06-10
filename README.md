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
* Biquad
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

## Tests

The tests are handled by the [Catch2](https://github.com/catchorg/Catch2.git) testing framework also defined as a Git submodule.

The accuracy tests use [Libtorch](https://pytorch.org/get-started/locally/) as a reference to verify numerical correctness. When configuring, pass the path to the Libtorch directory via `CMAKE_PREFIX_PATH`:

```shell
mkdir build && cd build
cmake .. -DNANOFLARE_TESTING=ON -DCMAKE_PREFIX_PATH=<path/to/libtorch>
cmake --build . --config Release 
```

Run accuracy tests:

```shell
make test
```

The `generate_tests_data.py` script generates the test data used by the accuracy tests. Install the Python dependencies first with [uv](https://github.com/astral-sh/uv), then run the script:

```shell
uv sync
uv run generate_tests_data.py
```

Benchmarks comparing Nanoflare against TorchScript/Libtorch are located in the `nanoflare_research` repository.


## Extending Nanoflare with Custom Models

Nanoflare uses static initialization to register models automatically. This allows you to add custom models in separate repositories without modifying nanoflare's code. This is useful for proprietary models or research projects.

### Creating a Private Model Repository

**1. Repository Structure:**

```
your-private-models/
├── CMakeLists.txt
├── models/
│   ├── YourModel.h              # C++ implementation
│   └── PrivateModels.h          # Registration header
├── python/
│   ├── your_model.py            # Python implementation
│   └── __init__.py
└── nanoflare/                   # Git submodule pointing to this repo
```

**2. C++ Model Registration:**

Create a header file that registers your models using static initialization:

```cpp
// models/PrivateModels.h
#pragma once

#include "nanoflare/ModelBuilder.h"
#include "YourModel.h"
#include "AnotherModel.h"

namespace YourNamespace
{
    namespace {
        inline bool registerPrivateModels()
        {
            // Register your custom models
            Nanoflare::registerModel<YourModel>("YourModel");
            Nanoflare::registerModel<AnotherModel>("AnotherModel");
            return true;
        }

        // Auto-register during static initialization (before main())
        static const bool _privateModelsRegistered = registerPrivateModels();
    }
}
```

**3. CMake Integration:**

In your private repository's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(YourPrivateModels)

# Add nanoflare as subdirectory (git submodule)
add_subdirectory(nanoflare)

# Create your models library
add_library(your_models INTERFACE)
target_link_libraries(your_models INTERFACE nanoflare)
target_include_directories(your_models INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)
```

**4. Using Custom Models in C++:**

Simply include your registration header before using `ModelBuilder`:

```cpp
#include "nanoflare/ModelBuilder.h"
#include "models/PrivateModels.h"  // Auto-registers via static initialization

// Now your models are registered and can be loaded from JSON
std::ifstream model_file("your_model.json");
std::shared_ptr<Nanoflare::BaseModel> model;
nlohmann::json j = nlohmann::json::parse(model_file);
Nanoflare::ModelBuilder::getInstance().buildModel(j, model);
```
