cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Venturi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark QUIET)
find_package(Vulkan REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(glm REQUIRED)

# ImGui & ImPlot
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/vendor/imgui)
set(IMPLOT_DIR ${CMAKE_SOURCE_DIR}/vendor/implot)

file(GLOB IMGUI_SOURCES 
    "${IMGUI_DIR}/*.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
)
file(GLOB IMPLOT_SOURCES "${IMPLOT_DIR}/*.cpp")

# Recursively add all source files in src/
file(GLOB_RECURSE VENTURI_SOURCES "src/*.cpp" "src/*.cu")
list(FILTER VENTURI_SOURCES EXCLUDE REGEX ".*/src/main\\.cpp$")

# Main library (shared between main executable and tests)
add_library(venturi_lib
    ${VENTURI_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

set_property(TARGET venturi_lib PROPERTY CUDA_SEPARABLE_COMPILATION OFF)

target_include_directories(venturi_lib PUBLIC 
    ${CMAKE_SOURCE_DIR}/src
    ${IMGUI_DIR}
    ${IMPLOT_DIR}
)
target_link_libraries(venturi_lib CUDA::cudart CUDA::curand Vulkan::Vulkan glfw glm::glm)

# Copy shaders to the build directory so executables can find them
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})

# Main executable
add_executable(venturi src/main.cpp)
target_link_libraries(venturi venturi_lib)
target_compile_options(venturi PRIVATE 
    $<$<CONFIG:Release>:-O2> 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe=--diag_suppress=20044>
)

# Testing
enable_testing()

# Discover ALL cpp files in tests/
file(GLOB_RECURSE ALL_TEST_FILES "tests/*.cpp")

# Filter OUT the benchmark files and profile files
set(TEST_SOURCES ${ALL_TEST_FILES})
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*tests/benchmark/.*")
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*tests/profile/.*")

# Create the Test Executable with the filtered list
add_executable(venturi_tests
    ${TEST_SOURCES}
)
target_link_libraries(venturi_tests venturi_lib GTest::gtest GTest::gtest_main)
target_include_directories(venturi_tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/tests
)

include(GoogleTest)
gtest_discover_tests(venturi_tests)

# Benchmarks (separate executable using Google Benchmark)
if(benchmark_FOUND)
    file(GLOB_RECURSE BENCHMARK_SOURCES "tests/benchmark/*.cpp")
    
    add_executable(venturi_benchmarks
        ${BENCHMARK_SOURCES}
    )
    target_link_libraries(venturi_benchmarks 
        venturi_lib 
        benchmark::benchmark
        benchmark::benchmark_main
    )
    target_include_directories(venturi_benchmarks PRIVATE 
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
    )
    # Disable LTO to avoid version mismatch with system benchmark library
    set_target_properties(venturi_benchmarks PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
    target_link_options(venturi_benchmarks PRIVATE -fno-lto)
endif()

# Profiling Tool
add_executable(venturi_profile
    tests/profile/profile_simulation.cpp
)
target_link_libraries(venturi_profile venturi_lib)
target_include_directories(venturi_profile PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)

# Max Agents Determination Tool
add_executable(venturi_max_agents
    tests/profile/determine_max_agents.cpp
)
target_link_libraries(venturi_max_agents venturi_lib)
target_include_directories(venturi_max_agents PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
)