#!/usr/bin/env bash
set -euo pipefail
mkdir -p reports/csv
for REP in reports/*.ncu-rep; do
  [ -e "$REP" ] || continue
  BASE=$(basename "$REP" .ncu-rep)
  ncu --import "$REP" --csv > "reports/csv/${BASE}.csv"
done
