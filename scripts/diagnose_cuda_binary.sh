#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR=${BUILD_DIR:-build}
BIN="$BUILD_DIR/gpu_cpu_optimization"

echo "== Toolchain =="
command -v nvcc || true
nvcc --version || true
nvidia-smi | sed -n '1,12p' || true

echo
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
  echo "== CMake CUDA architecture cache =="
  grep -E 'CUDA_ARCH|CMAKE_CUDA_ARCHITECTURES' "$BUILD_DIR/CMakeCache.txt" || true
fi

echo
if [ -f "$BIN" ]; then
  echo "== Embedded cubins/PTX =="
  cuobjdump --list-elf "$BIN" | grep -E 'sm_[0-9]+' || true
  echo
  echo "Expected for RTX 5060 Ti: sm_120 must appear above."
else
  echo "Binary not found: $BIN"
fi
