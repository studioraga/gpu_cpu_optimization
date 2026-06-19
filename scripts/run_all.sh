#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR=${BUILD_DIR:-build}
N=${N:-16777216}
CUDA_ARCH=${CUDA_ARCH:-120}
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_CPU_OPT_CUDA_ARCH="$CUDA_ARCH"
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$BUILD_DIR/gpu_cpu_optimization" all "$N" | tee results_all.txt
