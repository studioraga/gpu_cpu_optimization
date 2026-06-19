#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR=${BUILD_DIR:-build}
BIN="$BUILD_DIR/gpu_cpu_optimization"
mkdir -p reports
cuobjdump --dump-sass "$BIN" > reports/sass.txt
cuobjdump --dump-ptx "$BIN" > reports/ptx.txt
# Quick register/spill clues from build log are printed by -Xptxas=-v during build.
echo "Wrote reports/sass.txt and reports/ptx.txt"
