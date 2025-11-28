cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Venturi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA 11.5 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(venturi src/main.cpp)

target_link_libraries(venturi ${CUDA_LIBRARIES})