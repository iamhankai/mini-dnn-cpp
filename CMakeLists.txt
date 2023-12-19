cmake_minimum_required(VERSION 3.10)

project(Lenet5App LANGUAGES CXX CUDA)

# CUDA-specific settings
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add your CUDA and C++ source files
file(GLOB CUDA_SOURCES *.cu)
file(GLOB CPP_SOURCES *.cpp)

# Add executable and link CUDA libraries
cuda_add_executable(lenet5_app ${CUDA_SOURCES} ${CPP_SOURCES})
target_link_libraries(lenet5_app ${CUDA_LIBRARIES})
