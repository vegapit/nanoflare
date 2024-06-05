cmake_minimum_required( VERSION 3.24 )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

project( microtorch )

find_package(catch2 3 REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)
find_package(benchmark REQUIRED)
find_package(Torch REQUIRED PATHS "/Users/korchov/3rdparty/libtorch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_executable(regression_tests regression_tests.cpp)
target_link_libraries(
    regression_tests
    PUBLIC
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
)

add_executable(benchmark benchmark.cpp)
target_link_libraries(
    benchmark
    PUBLIC
    nlohmann_json::nlohmann_json
    benchmark::benchmark
    "${TORCH_LIBRARIES}"
)

include(CTest)
include(Catch)

enable_testing()

catch_discover_tests(regression_tests)