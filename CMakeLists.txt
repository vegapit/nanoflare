cmake_minimum_required( VERSION 3.24 )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

project( microtorch )

find_package(catch2 3 REQUIRED)
find_package(nlohmann_json 3.11.3 REQUIRED)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/include )

add_executable(lstm_test lstm_test.cpp)
target_link_libraries(
    lstm_test
    PUBLIC
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
)