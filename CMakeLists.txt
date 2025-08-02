cmake_minimum_required(VERSION 3.22)
project(example LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 90)

set(NVSHMEM_INCLUDE_DIR /usr/include/nvshmem_12)
set(NVSHMEM_LIB_DIR /usr/lib/x86_64-linux-gnu/nvshmem/12)

add_executable(example.out example.cu)

target_include_directories(example.out PRIVATE ${NVSHMEM_INCLUDE_DIR})

target_link_directories(example.out PRIVATE ${NVSHMEM_LIB_DIR})

target_link_libraries(example.out PRIVATE
  nvshmem
  nvidia-ml
  cuda
  cudart
)

