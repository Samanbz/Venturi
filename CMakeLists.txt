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


# Main library (shared between main executable and tests)
add_library(venturi_lib
    src/simulation.cpp
    src/simulation.cu
    src/canvas.cpp
)
target_include_directories(venturi_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(venturi_lib CUDA::cudart CUDA::curand Vulkan::Vulkan glfw)

# Main executable
add_executable(venturi src/main.cpp)
target_link_libraries(venturi venturi_lib)
target_compile_options(venturi PRIVATE $<$<CONFIG:Release>:-O2>)
# Testing
enable_testing()

# Discover ALL cpp files in tests/
file(GLOB_RECURSE ALL_TEST_FILES "tests/*.cpp")

# Filter OUT the benchmark files
set(TEST_SOURCES ${ALL_TEST_FILES})
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*tests/benchmark/.*")

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