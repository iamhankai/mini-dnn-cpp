cmake_minimum_required (VERSION 3.0)

project (Lenet5Pro LANGUAGES CXX CUDA)

# CUDA-specific settings
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories("${PROJECT_SOURCE_DIR}/third_party/eigen")


# Add your CUDA files
file(GLOB CUDA_SOURCES *.cu)

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

target_link_libraries(lenet5_pro MiniDNNLib)
target_link_libraries(lenet5_pro ${CUDA_LIBRARIES})

cuda_add_executable(lenet5_pro ${CUDA_SOURCES} ${CPP_SOURCES})