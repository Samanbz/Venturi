cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Venturi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA 11.5 REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark QUIET)

include_directories(${CUDA_INCLUDE_DIRS})

# Main library (shared between main executable and tests)
add_library(venturi_lib
    src/simulation.cpp
    src/simulation.cu
)
target_include_directories(venturi_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(venturi_lib ${CUDA_LIBRARIES} curand)

# Main executable
add_executable(venturi src/main.cpp)
target_link_libraries(venturi venturi_lib)

# Testing
enable_testing()

# Automatically discover unit test files (excluding benchmarks)
file(GLOB_RECURSE TEST_SOURCES "tests/unit/*.cpp")
file(GLOB_RECURSE FIXTURE_SOURCES "tests/fixtures/*.cpp")
# Exclude benchmark fixture from unit tests
list(FILTER FIXTURE_SOURCES EXCLUDE REGEX ".*simulation_benchmark_fixture\\.cpp$")

add_executable(venturi_tests
    ${FIXTURE_SOURCES}
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
        tests/fixtures/simulation_benchmark_fixture.cpp
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