add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/json)

add_library(microtorch INTERFACE)
target_link_libraries(
    microtorch
    INTERFACE
    nlohmann_json::nlohmann_json
)
target_include_directories(
    microtorch
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
)