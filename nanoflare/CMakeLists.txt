add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/json)

add_library(nanoflare INTERFACE)
target_link_libraries(
    nanoflare
    INTERFACE
    nlohmann_json::nlohmann_json
)
target_include_directories(
    nanoflare
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen
)