find_package(Torch REQUIRED PATHS ${LIBTORCH_DIR})

# Submodules
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/catch2)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/fmt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(layers_accuracy layers_accuracy.cpp)
target_link_libraries(
    layers_accuracy
    PRIVATE
    nanoflare
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
    "${TORCH_LIBRARIES}"
)
target_compile_options(layers_accuracy PRIVATE -march=native)
add_test(NAME layers_accuracy COMMAND layers_accuracy)

add_executable(models_accuracy models_accuracy.cpp)
target_link_libraries(
    models_accuracy
    PRIVATE
    nanoflare
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
    "${TORCH_LIBRARIES}"
)
target_compile_options(models_accuracy PRIVATE -march=native)
add_test(NAME models_accuracy COMMAND models_accuracy)

add_executable(models_benchmarking models_benchmarking.cpp)
target_link_libraries(
    models_benchmarking
    PRIVATE
    nanoflare
    Catch2::Catch2WithMain
    nlohmann_json::nlohmann_json
    "${TORCH_LIBRARIES}"
)
target_compile_options(models_benchmarking PRIVATE -march=native)