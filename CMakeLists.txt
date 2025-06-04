cmake_minimum_required( VERSION 3.24 )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG") # For GCC/Clang

project( nanoflare )

add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_subdirectory( nanoflare )

include(CTest)
enable_testing()

add_subdirectory( tests )