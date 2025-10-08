find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# Submodules
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)

# Python extension module
pybind11_add_module(nanoflare_py nanoflare_py.cpp)

target_link_libraries(nanoflare_py PRIVATE nanoflare)
target_include_directories(nanoflare_py PRIVATE ${NANOFLARE_INCLUDE_DIRS})