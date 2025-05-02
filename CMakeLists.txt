cmake_minimum_required( VERSION 3.24 )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

project( microtorch )

find_package(catch2 3 REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(Torch REQUIRED PATHS "/Users/korchov/3rdparty/libtorch")

# fmt
add_subdirectory("/Users/korchov/3rdparty/fmt" ${CMAKE_BINARY_DIR}/fmt)

# Nanobench
add_subdirectory("/Users/korchov/3rdparty/nanobench" ${CMAKE_BINARY_DIR}/nanobench)

# Eigen 3
include_directories("/Users/korchov/3rdparty/eigen-3.4.0/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${CMAKE_SOURCE_DIR}/include)

include(CTest)
enable_testing()

add_executable(layers_accuracy src/layers_accuracy.cpp)
target_link_libraries(
    layers_accuracy
    PUBLIC
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
    "${TORCH_LIBRARIES}"
)
add_test(NAME layers_accuracy COMMAND layers_accuracy)

add_executable(models_accuracy src/models_accuracy.cpp)
target_link_libraries(
    models_accuracy
    PUBLIC
    nlohmann_json::nlohmann_json
    Catch2::Catch2WithMain
    "${TORCH_LIBRARIES}"
)
add_test(NAME models_accuracy COMMAND models_accuracy)

add_executable(models_benchmarking src/models_benchmarking.cpp)
target_link_libraries(
    models_benchmarking
    PUBLIC
    fmt::fmt
    nanobench
    nlohmann_json::nlohmann_json
    "${TORCH_LIBRARIES}"
)