# GPU/CPU Optimization Lab for Node1

This lab converts all topics into runnable CUDA/C++17 examples:

- CPU scalar vs parallel memory-bandwidth baseline
- GPU coalesced vector add: streaming, memory-bound, L2 hit-rate may be misleading
- Strided/uncoalesced vector add: bad memory transactions
- Occupancy/register pressure: high occupancy is not automatically high performance
- Reduction: bad atomic reduction vs block/warp reduction
- Matrix transpose: naive vs tiled shared-memory padded version
- Shared-memory bank conflict: conflict vs padding
- AoS vs SoA layout: data layout and coalescing
- Pageable vs pinned host-to-device copy: full-pipeline bottleneck outside the kernel

## Build on Node1

```bash
Quick build steps, check and run test

source ~/.config/cuda-13.3.env

rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_CPU_OPT_CUDA_ARCH=120

cuobjdump --list-elf build/gpu_cpu_optimization | grep sm_

grep -E "CUDA_ARCH|CMAKE_CUDA_ARCHITECTURES" build/CMakeCache.txt

nvidia-smi
nvcc --version
cat /proc/driver/nvidia/version

./build/gpu_cpu_optimization all

GPU: NVIDIA GeForce RTX 5060 Ti
Compute capability: 12.0
SM count: 36
Global memory GiB: 15.47
Warp size: 32

cpu_vector_add_scalar              time_ms=9.4878     GB/s=21.22      GOP/s=1.77       status=PASS
cpu_vector_add_parallel            time_ms=9.2564     GB/s=21.75      GOP/s=1.81       status=PASS
gpu_vector_add_coalesced           time_ms=0.5101     GB/s=394.67     GOP/s=32.89      status=PASS
gpu_vector_add_strided             time_ms=0.6976     GB/s=36.08      GOP/s=3.01       status=PASS
low_register_occupancy_demo        time_ms=0.3519     GB/s=0.00       GOP/s=47.68      status=PASS
high_register_occupancy_demo       time_ms=1.5102     GB/s=0.00       GOP/s=11.11      status=PASS
reduce_atomic_bad                  time_ms=1.3025     GB/s=51.52      GOP/s=12.88      status=PASS
reduce_block_warp                  time_ms=0.1623     GB/s=413.58     GOP/s=103.37     status=PASS
transpose_naive                    time_ms=0.4483     GB/s=299.42     GOP/s=37.43      status=PASS
transpose_tiled_padded             time_ms=0.3666     GB/s=366.14     GOP/s=45.77      status=PASS
shared_bank_conflict               time_ms=0.0041     GB/s=0.00       GOP/s=0.00       status=PASS
shared_bank_padded                 time_ms=0.0041     GB/s=0.00       GOP/s=0.00       status=PASS
aos_update_x                       time_ms=2.0910     GB/s=96.28      GOP/s=8.02       status=PASS
soa_update_x                       time_ms=0.5078     GB/s=396.45     GOP/s=33.04      status=PASS
h2d_copy_pageable                  time_ms=6.8733     GB/s=9.76       GOP/s=0.00       status=PASS
h2d_copy_pinned                    time_ms=4.4685     GB/s=15.02      GOP/s=0.00       status=PASS

```bash

source ~/.config/cuda-13.3.env  # if you use this in your Node1 setup
cd gpu_cpu_optimization
./scripts/run_all.sh
```

Override architecture if needed:

```bash
CUDA_ARCH=120 ./scripts/run_all.sh
```

## Profile with Nsight Compute

```bash
./scripts/profile_ncu.sh
./scripts/export_ncu_csv.sh
```

Reports are written under `reports/`.

## Quick profiler interpretation

| Case | Expected bottleneck | Metrics to inspect |
|---|---|---|
| vector | DRAM/L2 bandwidth; little reuse | memory throughput, memory transactions, L2 hit rate context |
| stride | uncoalesced global memory | sectors/transactions per request, L2/DRAM throughput, stall reasons |
| occupancy | register pressure / latency hiding | registers per thread, achieved occupancy, eligible warps, spills |
| reduction | atomic contention vs staged reduction | atomic stalls, synchronization, memory throughput |
| transpose | uncoalesced writes fixed by tiling | global memory coalescing, shared bank conflicts |
| bank | shared-memory conflicts | shared load/store conflict metrics, elapsed time |
| layout | AoS stride vs SoA contiguous access | memory transactions, useful bandwidth |
| copy | host-device transfer overhead | memcpy throughput, pageable vs pinned transfer |

## Explanation flow

For each case, answer in this pattern:

1. Baseline: what the kernel does.
2. Hypothesis: compute-bound, memory-bound, cache-bound, occupancy-bound, divergence-bound, atomic-bound, copy-bound.
3. Profiler evidence: which metrics support or reject the hypothesis.
4. Code change: what you changed.
5. Validation: correctness PASS and before/after time.
6. Metric caution: which metric can mislead.
