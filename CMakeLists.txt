cmake_minimum_required( VERSION 3.24 )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "CMAKE_BUILD_TYPE Type Unspecified; Release by default")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

project( nanoflare )

# Option defining test builds
option(NANOFLARE_TESTING "Build tests and benchmarks?" OFF)

# Add 3rdParty dependencies
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/eigen)
add_subdirectory(${CMAKE_SOURCE_DIR}/libs/json)

add_library(nanoflare INTERFACE)
target_link_libraries(
    nanoflare
    INTERFACE
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
)
target_include_directories(
    nanoflare
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(${NANOFLARE_TESTING})
    include(CTest)
    enable_testing()
    
    # Add this path definition for test code
    add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    add_subdirectory( tests )
endif()