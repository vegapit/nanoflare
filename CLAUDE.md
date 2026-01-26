# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Nanoflare is a header-only C++17 library for fast, lightweight real-time inference of PyTorch models, originally developed for audio plugins. It consists of two parallel implementations:

- **Python library** (`pynanoflare/`): PyTorch wrapper modules for model calibration and export
- **C++ library** (`include/nanoflare/`): Header-only inference engine using Eigen3 for matrix operations

The workflow involves training/calibrating in Python, serializing to JSON, and loading in C++ for high-performance inference.

## Build Commands

### Standard C++ library build
```bash
mkdir build && cd build
cmake ..
cmake --build .
```

### Build with tests (requires LibTorch)
```bash
mkdir build && cd build
cmake .. -DNANOFLARE_TESTING=ON -DLIBTORCH_DIR=<path/to/libtorch>
cmake --build .
```

### Run tests
```bash
# From build directory
make test

# Or run individual test executables
./tests/layers_accuracy
./tests/models_accuracy
```

### Run benchmarks
```bash
# From build directory
./tests/models_benchmarking
```

### Python Bindings

Python bindings have been moved to a separate repository: **nanoflare-python**

This keeps the core C++ library independent and allows users to create their own bindings with custom models.

### Install Python package
```bash
pip install -e .
```

The `pynanoflare` package provides PyTorch wrappers for creating and exporting models.

## Architecture

### Model Registration System

The core architectural pattern uses runtime registration via `ModelBuilder` singleton:

1. **C++ side**: Models inherit from `BaseModel` and provide a static `build()` function
2. **Registration**: `registerModel<T>("ModelName")` adds model to the builder registry
3. **Python export**: Models implement `generate_doc()` to serialize weights to JSON
4. **C++ import**: `ModelBuilder::buildModel()` deserializes JSON and constructs model

**Static Initialization for Extensibility**:
- Models auto-register via static initialization (executes before `main()`)
- `BuiltinModels.h` registers nanoflare's built-in models automatically
- External projects create their own registration headers with the same pattern
- No modifications to nanoflare code needed for private/external models

This allows new models to be defined in separate repositories while remaining discoverable. See README.md "Extending Nanoflare with Custom Models" for details on creating private model repositories.

### Layer and Model Hierarchy

- **Base layers** (`include/nanoflare/layers/`): BatchNorm1d, Conv1d, GRU, LSTM, Linear, PReLU
- **Custom blocks**: CausalDilatedConv1d, FiLM, MicroTCNBlock, ResidualBlock, TCNBlock, PlainSequential
- **Complete models** (`include/nanoflare/models/`): HammersteinWiener, MicroTCN, ResRNN (ResGRU/ResLSTM), TCN, WaveNet

All models inherit from `BaseModel` which provides:
- `forward()`: Pure virtual method for inference
- `loadStateDict()`: Pure virtual method for loading weights from JSON
- `normalise()/denormalise()`: Input/output normalization utilities
- Static `build()`: Factory method for `ModelBuilder` registration

### Python-C++ Correspondence

Each model exists in both implementations:

- **Python** (`pynanoflare/`): Inherits from `BaseModel(nn.Module)`, implements `generate_doc()` for JSON export
- **C++** (`include/nanoflare/models/`): Inherits from `BaseModel`, implements `build()` for JSON import

Layers follow the same pattern. Custom layers implement `generate_doc()` in Python and corresponding deserialization in C++.

### Test Data Generation

The file `tests/data/` contains JSON models and TorchScript files for accuracy testing. Test data is generated using Python scripts that export calibrated models to both formats for comparison.

## Adding New Models

To add a new neural network architecture:

1. **Python class** in `pynanoflare/`: Define module inheriting from `BaseModel`, implement `generate_doc()` for JSON serialization
2. **C++ header** in `include/nanoflare/models/`: Define class inheriting from `BaseModel`, implement:
   - `forward()`: Inference logic
   - `loadStateDict()`: Weight loading from JSON
   - Static `build()`: Factory method that parses JSON and constructs model
3. **Register model**: Add to `BuiltinModels.h` for builtin models, or create your own registration header for private models (see README.md "Extending Nanoflare")

Reference existing models (e.g., `MicroTCN.h` and `pynanoflare/tcn.py`) for implementation patterns.

**For Python bindings**: See the separate **nanoflare-python** repository for creating Python bindings with pybind11.

## Key Files

- `include/nanoflare/ModelBuilder.h`: Singleton registry for model factories
- `include/nanoflare/BuiltinModels.h`: Auto-registration of built-in models via static initialization
- `include/nanoflare/models/BaseModel.h`: Abstract base class for all models
- `include/nanoflare/Functional.h`: Activation functions and operators
- `include/nanoflare/utils.h`: Type aliases and utilities
- `pynanoflare/modules.py`: Base classes and custom layers for Python (training/export)

## Dependencies

- **Eigen3**: Linear algebra library (git submodule at `libs/eigen/`)
- **nlohmann/json**: JSON parsing (git submodule at `libs/json/`)
- **Catch2**: Testing framework (git submodule at `tests/Catch2/`)
- **fmt**: Formatting library for tests (git submodule at `tests/fmt/`)
- **LibTorch**: Required only for testing (compare against PyTorch reference)

All submodules are automatically built with the library.

**For Python development:**
- **PyTorch**: Required by `pynanoflare` for training and exporting models
- **pybind11**: Required by **nanoflare-python** (separate repo) for Python bindings

## Using as Git Submodule

To integrate Nanoflare into another project:

```cmake
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nanoflare)
include_directories(${NANOFLARE_INCLUDE_DIRS})
```

The `NANOFLARE_INCLUDE_DIRS` variable includes Nanoflare headers and dependencies (Eigen, nlohmann/json).
