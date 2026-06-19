#!/usr/bin/env bash
set -euo pipefail
BUILD_DIR=${BUILD_DIR:-build}
BIN="$BUILD_DIR/gpu_cpu_optimization"
N=${N:-16777216}
mkdir -p reports

# Use --kernel-name regex to keep reports focused. Metric names can change across Nsight Compute versions,
# so --set full is the most portable first pass.
for CASE in vector stride occupancy reduction transpose bank layout copy; do
  echo "Profiling case: $CASE"
  ncu --set full \
      --target-processes all \
      --force-overwrite \
      -o "reports/ncu_${CASE}" \
      "$BIN" "$CASE" "$N" \
      > "reports/ncu_${CASE}.txt" 2>&1 || true
  echo "  saved reports/ncu_${CASE}.ncu-rep and reports/ncu_${CASE}.txt"
done
