cmake_minimum_required(VERSION 3.24)
project(gpu_cpu_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Node1 RTX 5060 Ti / Blackwell is compute capability 12.0.
#
# Important: CMake can pre-populate CMAKE_CUDA_ARCHITECTURES from the compiler
# default before this file reaches an `if(NOT DEFINED ...)` check. On this lab
# that default showed up as sm_75, which builds a Turing cubin plus PTX fallback.
# Running that binary on RTX 5060 Ti then forces PTX JIT and can fail with:
#   "the provided PTX was compiled with an unsupported toolchain"
#
# So this project uses its own cache variable and applies it directly to the
# target. Override only when intentionally building for another GPU:
#   cmake -S . -B build -DGPU_CPU_OPT_CUDA_ARCH=120
set(GPU_CPU_OPT_CUDA_ARCH "120" CACHE STRING "CUDA architecture for this lab, for example 120 for RTX 5060 Ti")

find_package(OpenMP)

add_executable(gpu_cpu_optimization
  src/gpu_cpu_optimization.cu
)

set_property(TARGET gpu_cpu_optimization PROPERTY CUDA_ARCHITECTURES ${GPU_CPU_OPT_CUDA_ARCH})

target_include_directories(gpu_cpu_optimization PRIVATE include)

target_compile_options(gpu_cpu_optimization PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gpu_cpu_optimization PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(gpu_cpu_optimization PRIVATE HAS_OPENMP=1)
endif()

message(STATUS "gpu_cpu_optimization CUDA arch = ${GPU_CPU_OPT_CUDA_ARCH}")
