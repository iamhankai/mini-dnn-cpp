cmake_minimum_required (VERSION 3.1)
project(myproject LANGUAGES CXX CUDA)  # Specify CUDA for the project

include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")

# Find CUDA package
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

aux_source_directory(. DIR_SRCS)
add_subdirectory(src)
file(GLOB_RECURSE CUDA_SOURCES "src/device/*.cu")

cuda_add_executable(main ${CUDA_SOURCES} ${DIR_SRCS})

target_link_libraries(main MiniDNNLib)
target_link_libraries(main DeviceLib)
target_link_libraries(main ${CUDA_LIBRARIES})