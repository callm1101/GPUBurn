cmake_minimum_required(VERSION 3.27)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "cmake --verbose on" FORCE)

set(COMPILER_PATH "/usr/bin;/usr/local/bin")
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler -ccbin /usr/bin/clang")

project(gpu_burn LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_LINKER "${LLD}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")

set(CMAKE_CUDA_ARCHITECTURES native)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

add_executable(gpu_burn src/gpu_burn.cu)

target_link_libraries(gpu_burn CUDA::cublas)
