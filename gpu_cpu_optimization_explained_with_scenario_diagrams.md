# GPU/CPU Optimization Lab — detailed code explanation

This note explains your attached files line by line and connects each code section to  scenarios, validation, and profiling.

Files covered:

- `cuda_check.hpp`
- `gpu_cpu_optimization.cu`

The code is structured as a practical  lab. It does not only show CUDA syntax; it teaches how to explain **why** a kernel is slow or fast: memory bandwidth, coalescing, occupancy, register pressure, atomics, warp reduction, shared-memory bank conflicts, data layout, and host-device copy overhead.

---

## Build on Node1

For your Node1 CUDA 13.3 / RTX 5060 Ti style setup:

```bash
source ~/.config/cuda-13.3.env

mkdir -p build reports

nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
     -Xptxas=-v \
     -I. \
     gpu_cpu_optimization.cu \
     -o build/gpu_cpu_optimization
```

With OpenMP enabled for CPU parallel baseline:

```bash
nvcc -std=c++17 -O3 -lineinfo -arch=sm_120 \
     -Xptxas=-v \
     -Xcompiler -fopenmp \
     -DHAS_OPENMP \
     -I. \
     gpu_cpu_optimization.cu \
     -o build/gpu_cpu_optimization \
     -lgomp
```

Run all tests:

```bash
./build/gpu_cpu_optimization all 16777216
```

---

## Top-level execution flow

```text
main()
  |
  +-- print_device_info()
  |
  +-- cpu_vector_add_scalar() / cpu_vector_add_parallel()
  |
  +-- run_gpu_vector_add()
  |     +-- vector_add_coalesced_kernel()
  |     +-- vector_add_strided_kernel()
  |
  +-- run_occupancy()
  |     +-- low_register_kernel()
  |     +-- high_register_kernel()
  |
  +-- run_reduction()
  |     +-- reduce_atomic_bad_kernel()
  |     +-- reduce_block_warp_kernel()
  |
  +-- run_transpose()
  |     +-- transpose_naive_kernel()
  |     +-- transpose_tiled_kernel()
  |
  +-- run_bank_conflict()
  |     +-- shared_bank_conflict_kernel()
  |     +-- shared_bank_padded_kernel()
  |
  +-- run_aos_soa()
  |     +-- update_aos_kernel()
  |     +-- update_soa_kernel()
  |
  +-- run_copy()
        +-- pageable H2D copy
        +-- pinned H2D copy
```

---


## Text-view diagrams for discussion and  explanation

This section adds discussion-ready text diagrams. Use these diagrams when you need to explain the lab verbally to an er, reviewer, or teammate. The goal is to show not only **what the code does**, but also **how data moves**, **where time is spent**, **what profiler metric matters**, and **which optimization decision follows from the evidence**.

---

### 0. Complete Node1 lab architecture view

```text
                          Node1: CUDA/C++17 GPU Optimization Lab
┌──────────────────────────────────────────────────────────────────────────────┐
│ Host CPU side                                                                 │
│                                                                              │
│  main()                                                                       │
│    │                                                                         │
│    ├── parse command: all/cpu/vector/stride/occupancy/reduction/...          │
│    │                                                                         │
│    ├── allocate std::vector input/output/reference buffers                    │
│    │                                                                         │
│    ├── fill deterministic test data                                           │
│    │                                                                         │
│    ├── compute CPU reference where needed                                     │
│    │                                                                         │
│    ├── allocate CUDA memory using RAII wrappers                               │
│    │       DeviceBuffer<T>        -> cudaMalloc/cudaFree                      │
│    │       PinnedHostBuffer<T>    -> cudaMallocHost/cudaFreeHost              │
│    │                                                                         │
│    ├── copy input to GPU                                                      │
│    │       cudaMemcpy HostToDevice                                            │
│    │                                                                         │
│    ├── launch CUDA kernel                                                     │
│    │       <<<grid, block>>>                                                  │
│    │                                                                         │
│    ├── time kernel with CUDA events                                           │
│    │       GpuTimer.start() / stop_ms()                                       │
│    │                                                                         │
│    ├── copy output back                                                       │
│    │       cudaMemcpy DeviceToHost                                            │
│    │                                                                         │
│    ├── validate correctness                                                   │
│    │       approx_equal() / finite check / CPU reference compare              │
│    │                                                                         │
│    └── print benchmark result                                                 │
│            name, time_ms, GB/s, GOP/s, PASS/FAIL                             │
│                                                                              │
├────────────────────────────────────── PCIe / Host-device boundary ───────────┤
│                                                                              │
│ GPU device side                                                               │
│                                                                              │
│  Global memory / VRAM                                                         │
│      │                                                                       │
│      ├── input buffers                                                        │
│      ├── output buffers                                                       │
│      └── partial reduction buffers                                            │
│                                                                              │
│  L2 cache                                                                     │
│      │                                                                       │
│      └── all SMs share this cache path                                        │
│                                                                              │
│  Streaming Multiprocessors / SMs                                              │
│      │                                                                       │
│      ├── blocks assigned to SMs                                               │
│      ├── warps execute inside blocks                                          │
│      ├── registers hold per-thread live variables                             │
│      ├── shared memory holds per-block tile/reduction data                    │
│      └── CUDA cores execute arithmetic and address-generation work            │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                         Nsight Compute / ncu reports
          kernel time | occupancy | registers | L2/DRAM throughput |
          memory transactions | bank conflicts | warp stall reasons
```

How to explain:

```text
I built the lab so each case follows a controlled experimental pattern:
known input -> GPU execution -> validation -> timing -> profiler evidence.
This allows me to discuss optimization as an evidence-driven process, not guesswork.
```

---

### 1. Benchmark and validation control loop

```text
For each benchmark case:

┌───────────────┐
│ Select case   │  cpu/vector/stride/occupancy/reduction/transpose/bank/layout/copy
└───────┬───────┘
        │
        ▼
┌───────────────────┐
│ Generate input     │  deterministic random data for repeatability
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ CPU reference      │  expected output or expected scalar checksum
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ GPU allocation     │  DeviceBuffer<T> / PinnedHostBuffer<T>
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Warm-up kernel     │  removes first-launch overhead from timing
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Timed iterations   │  CUDA events measure GPU elapsed time
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Copy result back   │  only after timed region unless copy test
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Validate output    │  PASS/FAIL; never trust speed without correctness
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Report metrics     │  time_ms, useful GB/s, GOP/s, status
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Profile with ncu    │  confirm why the timing changed
└───────────────────┘
```

Discussion point:

```text
A fast wrong kernel is useless. This is why correctness validation is part of
the benchmark path and not an afterthought.
```

---

### 2. CUDA execution hierarchy used by the kernels

```text
CUDA launch syntax:
    kernel<<<grid, block>>>(...)

Example:
    grid  = many thread blocks
    block = 256 threads

┌──────────────────────────────────────────────────────────────┐
│ Grid                                                         │
│                                                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │
│  │ Block 0       │  │ Block 1       │  │ Block 2       │ ... │
│  │ 256 threads   │  │ 256 threads   │  │ 256 threads   │     │
│  └──────┬────────┘  └──────┬────────┘  └──────┬────────┘     │
│         │                  │                  │              │
└─────────┼──────────────────┼──────────────────┼──────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌──────────────────────────────────────────────────────────────┐
│ Streaming Multiprocessors / SMs                              │
│                                                              │
│  SM0 runs some blocks        SM1 runs some blocks       ...  │
│                                                              │
│  Inside one block:                                             │
│                                                              │
│    256 threads = 8 warps                                      │
│                                                              │
│    Warp 0: lanes  0..31                                       │
│    Warp 1: lanes 32..63                                       │
│    Warp 2: lanes 64..95                                       │
│    ...                                                        │
│    Warp 7: lanes 224..255                                     │
└──────────────────────────────────────────────────────────────┘
```

Global thread index formula used repeatedly in the code:

```text
i = blockIdx.x * blockDim.x + threadIdx.x

Example:
blockDim.x = 256

Block 0, thread 0    -> i = 0
Block 0, thread 255  -> i = 255
Block 1, thread 0    -> i = 256
Block 1, thread 255  -> i = 511
```

Grid-stride loop:

```text
stride = blockDim.x * gridDim.x

Thread starts at i and then processes:
    i
    i + stride
    i + 2*stride
    i + 3*stride
    ...

This allows a limited grid to process a very large input.
```

---

### 3. CPU baseline data-path diagram

Command:

```bash
./build/gpu_cpu_optimization cpu 16777216
```

```text
CPU scalar version:

CPU core
  │
  ├── load a[i] from DRAM/cache
  ├── load b[i] from DRAM/cache
  ├── add
  └── store c[i] to DRAM/cache

For N elements:

a[0] a[1] a[2] ... a[N-1]
  │    │    │         │
  ▼    ▼    ▼         ▼
b[0] b[1] b[2] ... b[N-1]
  │    │    │         │
  ▼    ▼    ▼         ▼
c[0] c[1] c[2] ... c[N-1]
```

OpenMP version:

```text
                 CPU socket / CPU cores
┌────────────┬────────────┬────────────┬────────────┐
│ Core 0     │ Core 1     │ Core 2     │ Core 3     │ ...
│ chunk 0    │ chunk 1    │ chunk 2    │ chunk 3    │
└─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘
      │            │            │            │
      ▼            ▼            ▼            ▼
  DRAM/cache bandwidth becomes the shared bottleneck
```

 explanation:

```text
This demonstrates that CPU parallelism is not infinite. Once memory bandwidth
is saturated, adding more CPU threads may not improve throughput. This matters
in AI pipelines where CPU preprocessing can dominate before data reaches the GPU.
```

---

### 4. Coalesced GPU vector-add diagram

Command:

```bash
./build/gpu_cpu_optimization vector 16777216
```

Kernel concept:

```text
c[i] = a[i] + b[i]
```

Warp memory access pattern:

```text
One warp = 32 lanes

Lane:       0    1    2    3    4    5          31
Thread i:   i   i+1  i+2  i+3  i+4  i+5  ...   i+31

a access:  a[i] a[i+1] a[i+2] ...              a[i+31]
b access:  b[i] b[i+1] b[i+2] ...              b[i+31]
c store:   c[i] c[i+1] c[i+2] ...              c[i+31]

Memory layout:
a:  ┌────┬────┬────┬────┬────┬────┬────┐
    │a[i]│a+1 │a+2 │a+3 │a+4 │... │a+31│
    └────┴────┴────┴────┴────┴────┴────┘
             contiguous memory
```

Hardware result:

```text
Adjacent lanes access adjacent floats
        │
        ▼
Coalesced global memory transactions
        │
        ▼
High useful bandwidth
        │
        ▼
Kernel usually memory-bandwidth-bound, not compute-bound
```

Profiler interpretation:

```text
Useful metrics:
  - DRAM throughput
  - L2 throughput
  - memory workload analysis
  - global load/store efficiency
  - warp stalls caused by memory dependency

Potentially misleading metric:
  - L2 hit rate

Why L2 hit rate may mislead:
  This kernel streams data once. There is little reuse. A low L2 hit rate may be
  expected and does not automatically indicate a bad kernel.
```

---

### 5. Strided / uncoalesced memory access diagram

Command:

```bash
./build/gpu_cpu_optimization stride 16777216
```

Kernel concept:

```text
j = i * stride_elems
c[j] = a[j] + b[j]
```

For `stride_elems = 8`:

```text
One warp = 32 lanes

Lane:        0      1      2      3      4          31
Logical i:   i     i+1    i+2    i+3    i+4   ...  i+31
Physical j:  i*8  (i+1)*8 (i+2)*8 ...              (i+31)*8

a access:
a[0]        a[8]        a[16]       a[24]       ... a[248]
 │           │           │           │               │
 ▼           ▼           ▼           ▼               ▼
far-apart memory locations, not adjacent locations
```

Comparison with coalesced:

```text
Coalesced:
lane 0 -> a[0]
lane 1 -> a[1]
lane 2 -> a[2]
lane 3 -> a[3]

Strided:
lane 0 -> a[0]
lane 1 -> a[8]
lane 2 -> a[16]
lane 3 -> a[24]
```

Hardware effect:

```text
Far-apart addresses per warp
        │
        ▼
More memory sectors / transactions
        │
        ▼
Lower useful GB/s
        │
        ▼
Same math, worse memory behavior
```

 explanation:

```text
The arithmetic did not change. Only the memory address pattern changed.
This is a clean demonstration that GPU performance is often controlled by
memory layout and coalescing, not by the number of arithmetic instructions.
```

---

### 6. Occupancy and register-pressure diagram

Command:

```bash
./build/gpu_cpu_optimization occupancy 16777216
```

Resource model per SM:

```text
One SM has finite resources:

┌────────────────────────────────────────────┐
│ Streaming Multiprocessor / SM              │
│                                            │
│  registers      : limited pool             │
│  shared memory  : limited pool             │
│  warp slots     : limited pool             │
│  block slots    : limited pool             │
│  schedulers     : issue warp instructions  │
└────────────────────────────────────────────┘
```

Low-register kernel:

```text
Each thread uses fewer live variables
        │
        ▼
More warps/blocks may fit on one SM
        │
        ▼
Higher theoretical occupancy possible
```

High-register kernel:

```text
Each thread keeps many variables alive:
r0, r1, r2, ... r15
        │
        ▼
More registers consumed per thread
        │
        ▼
Fewer warps/blocks may fit on one SM
        │
        ▼
Occupancy may drop
        │
        ▼
If register demand is too high, compiler may spill to local memory
```

Important caveat:

```text
High occupancy does not automatically mean high performance.

Occupancy helps hide latency:
  memory load pending -> scheduler runs another ready warp

But if the kernel is already bandwidth-saturated:
  adding more resident warps may not improve runtime

If a lower-occupancy kernel has enough instruction-level parallelism:
  it may still perform well
```

Profiler decision:

```text
Check together:
  - achieved occupancy
  - theoretical occupancy
  - registers per thread
  - local memory / spills
  - eligible warps per scheduler
  - warp stall reasons
  - SM throughput
```

 sentence:

```text
I treat occupancy as a diagnostic signal, not a goal by itself.
```

---

### 7. Reduction: bad global atomic vs staged warp/block reduction

Command:

```bash
./build/gpu_cpu_optimization reduction 16777216
```

Bad atomic design:

```text
Each thread computes a partial sum
        │
        ▼
Every thread or many threads call:
    atomicAdd(out, sum)
        │
        ▼
All updates target the same global address
        │
        ▼
Contention / serialization
        │
        ▼
Poor scaling
```

Diagram:

```text
Thread 0 ─┐
Thread 1 ─┤
Thread 2 ─┤
Thread 3 ─┤
...       ├── atomicAdd(global_out)
Thread N ─┘
```

Optimized staged design:

```text
Stage 1: each thread accumulates multiple elements
        │
        ▼
Stage 2: reduce within each warp using shuffle
        │
        ▼
Stage 3: lane 0 of each warp writes one value to shared memory
        │
        ▼
Stage 4: first warp reduces the warp sums
        │
        ▼
Stage 5: one partial sum written per block
```

Diagram:

```text
Input array
  │
  ▼
Grid-stride per-thread accumulation
  │
  ▼
┌──────────────────────── Block k ────────────────────────┐
│                                                          │
│  Warp 0: 32 lane sums -> warp_reduce_sum -> one sum       │
│  Warp 1: 32 lane sums -> warp_reduce_sum -> one sum       │
│  Warp 2: 32 lane sums -> warp_reduce_sum -> one sum       │
│  ...                                                     │
│                                                          │
│  shared memory: warp_sums[0..num_warps-1]                │
│                                                          │
│  First warp reduces warp_sums[]                          │
│                                                          │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
                     partial[blockIdx.x]
```

Why this helps:

```text
Global atomics reduced from many updates to one output per block.
Most reduction work happens in registers and shared memory.
```

Metrics:

```text
Bad version:
  - atomic stalls
  - serialization
  - higher global memory contention

Optimized version:
  - fewer global updates
  - some synchronization cost
  - better scalability
```

---

### 8. Matrix transpose: naive vs tiled padded

Command:

```bash
./build/gpu_cpu_optimization transpose
```

Naive transpose:

```text
Input matrix A[row][col]
Output matrix B[col][row]

B[x][y] = A[y][x]
```

Naive access pattern:

```text
Read may be coalesced:
A row-major:
A[y][x+0], A[y][x+1], A[y][x+2], ... are adjacent

But write may be strided:
B[x+0][y], B[x+1][y], B[x+2][y], ... are far apart in row-major layout
```

Diagram:

```text
Input A, row-major
┌────┬────┬────┬────┐
│A00 │A01 │A02 │A03 │  contiguous row
├────┼────┼────┼────┤
│A10 │A11 │A12 │A13 │
├────┼────┼────┼────┤
│A20 │A21 │A22 │A23 │
└────┴────┴────┴────┘

Output B = transpose(A)
┌────┬────┬────┐
│A00 │A10 │A20 │
├────┼────┼────┤
│A01 │A11 │A21 │
├────┼────┼────┤
│A02 │A12 │A22 │
├────┼────┼────┤
│A03 │A13 │A23 │
└────┴────┴────┘
```

Tiled transpose:

```text
Step 1: coalesced load from global memory into shared tile

Global A tile
        │ coalesced read
        ▼
Shared memory tile[32][33]
        │ transpose inside shared memory
        ▼
Global B tile
        │ coalesced write
        ▼
Output
```

Text view:

```text
┌──────────────────────┐
│ Global memory A       │
│ contiguous row reads  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Shared memory tile    │
│ tile[32][33]          │
│ +1 padding avoids     │
│ bank-conflict pattern │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Global memory B       │
│ coalesced row writes  │
└──────────────────────┘
```

Why `tile[32][33]`:

```text
tile[32][32] may cause column access bank conflicts.
tile[32][33] shifts each row by one extra float.
That changes address-to-bank mapping and reduces conflict.
```

 sentence:

```text
Transpose shows how shared-memory tiling can convert poor global access into
coalesced global access, while padding controls shared-memory bank behavior.
```

---

### 9. Shared-memory bank conflict diagram

Command:

```bash
./build/gpu_cpu_optimization bank
```

Shared memory bank model, simplified:

```text
Shared memory banks:
Bank:     0    1    2    3    4        31
          │    │    │    │    │        │
Address:  0    1    2    3    4   ...  31
Address: 32   33   34   35   36   ...  63
Address: 64   65   66   67   68   ...  95
```

Conflict case:

```text
shared float tile[32][32]

Column-style access:
lane 0 -> tile[0][same_col]
lane 1 -> tile[1][same_col]
lane 2 -> tile[2][same_col]
...
lane31 -> tile[31][same_col]

Because row length is 32, these addresses can map to the same bank.
```

Diagram:

```text
tile[0][0]   -> bank 0
tile[1][0]   -> bank 0
tile[2][0]   -> bank 0
...
tile[31][0]  -> bank 0

Many lanes -> same bank -> serialization
```

Padded case:

```text
shared float tile[32][33]

tile[0][0]   -> bank 0
tile[1][0]   -> bank 1
tile[2][0]   -> bank 2
...
tile[31][0]  -> bank31

Many lanes -> different banks -> parallel service
```

 sentence:

```text
Padding does not change the algorithm. It changes address mapping so the
hardware can serve more shared-memory accesses in parallel.
```

---

### 10. AoS vs SoA data-layout diagram

Command:

```bash
./build/gpu_cpu_optimization layout 16777216
```

AoS layout:

```text
struct ParticleAoS {
    float x, y, z, vx, vy, vz;
};

Memory:
P0: x y z vx vy vz | P1: x y z vx vy vz | P2: x y z vx vy vz | ...
```

When the kernel only updates x using vx:

```text
lane 0 reads P0.x and P0.vx
lane 1 reads P1.x and P1.vx
lane 2 reads P2.x and P2.vx

Memory addresses for x:
P0.x      P1.x      P2.x      P3.x
 │         │         │         │
 ▼         ▼         ▼         ▼
separated by full struct size, not contiguous floats
```

SoA layout:

```text
float x[N], y[N], z[N], vx[N], vy[N], vz[N];

Memory:
x:  x0  x1  x2  x3  x4  ... contiguous
vx: vx0 vx1 vx2 vx3 vx4 ... contiguous
```

When the kernel updates x using vx:

```text
lane 0 reads x[0], vx[0]
lane 1 reads x[1], vx[1]
lane 2 reads x[2], vx[2]

Adjacent lanes read adjacent memory
        │
        ▼
Better coalescing
```

Decision diagram:

```text
Kernel uses entire object per thread?
        │
        ├── yes -> AoS may be acceptable or simpler
        │
        └── no, only one/few fields across many objects
                 │
                 ▼
              SoA usually better for GPU coalescing
```

 sentence:

```text
Data layout is often the optimization. If the GPU kernel is field-wise, SoA
can improve coalescing before changing any arithmetic.
```

---

### 11. Pageable vs pinned host-device copy diagram

Command:

```bash
./build/gpu_cpu_optimization copy 16777216
```

Pageable copy path, simplified:

```text
Application pageable memory
        │
        │ cudaMemcpy HostToDevice
        ▼
Driver may stage through temporary pinned buffer
        │
        ▼
DMA transfer across PCIe
        │
        ▼
GPU global memory
```

Pinned copy path:

```text
Application pinned/page-locked memory
        │
        │ cudaMemcpy HostToDevice
        ▼
DMA transfer across PCIe
        │
        ▼
GPU global memory
```

Why it matters:

```text
Camera frame / tensor generated on CPU
        │
        ▼
Host-to-device copy
        │
        ▼
GPU kernel
        │
        ▼
Device-to-host copy or next GPU stage

If copy time dominates, kernel optimization alone will not fix end-to-end latency.
```

 sentence:

```text
For AI pipelines, I measure end-to-end time. I separate copy time from kernel
time because a fast kernel can still be hidden behind slow host-device movement.
```

---

### 12. Nsight Compute profiler decision tree

Use this after running:

```bash
ncu --set full --target-processes all --force-overwrite \
    -o reports/ncu_vector \
    ./build/gpu_cpu_optimization vector 16777216 \
    > reports/ncu_vector.txt 2>&1
```

Decision tree:

```text
Start with kernel time
        │
        ▼
Is SM busy high?
        │
        ├── yes
        │    │
        │    ▼
        │   Check compute pipeline / instruction mix / dependency stalls
        │
        └── no
             │
             ▼
        Is memory throughput high?
             │
             ├── yes
             │    │
             │    ▼
             │   Memory-bandwidth-bound
             │   Improve coalescing, reduce bytes, reuse data, change layout
             │
             └── no
                  │
                  ▼
             Are warp stalls high?
                  │
                  ├── memory dependency -> latency/coalescing/cache issue
                  ├── barrier          -> synchronization issue
                  ├── not selected     -> scheduling/occupancy interaction
                  ├── branch           -> divergence issue
                  └── execution dep    -> instruction dependency issue
```

Metric-to-action map:

```text
Metric observed                         Likely meaning                         Action
------------------------------------------------------------------------------------------------
Low useful GB/s + many memory sectors    uncoalesced global access              change layout/indexing
High DRAM throughput                     bandwidth-bound                        reduce bytes/reuse/tile
Low occupancy + high register count      register-limited occupancy             reduce live variables
Local memory/spills                      register spill                          simplify/fission/tune launch
High atomic stalls                       global contention                       reduce locally first
High barrier stalls                      synchronization cost                   reduce sync/fuse differently
Shared bank conflicts                    shared memory serialization            pad/rearrange shared layout
Low L2 hit for streaming kernel          may be normal                           do not chase blindly
High copy time vs kernel time            data movement bottleneck                pinned memory/batching/overlap
```

---

### 13.  storytelling diagram

Use this flow when discussing any one code case:

```text
Question from er
        │
        ▼
Name the workload pattern
        │
        ├── vector add       -> streaming memory-bound
        ├── stride           -> uncoalesced memory
        ├── occupancy        -> register pressure / latency hiding
        ├── reduction        -> atomics vs staged local reduction
        ├── transpose        -> tiling and coalescing
        ├── bank             -> shared memory bank mapping
        ├── layout           -> AoS vs SoA coalescing
        └── copy             -> end-to-end transfer overhead
        │
        ▼
State the bottleneck hypothesis
        │
        ▼
Point to profiler metrics
        │
        ▼
Explain the code change
        │
        ▼
Validate correctness
        │
        ▼
Compare before/after timing
        │
        ▼
Explain metric caveat
```

Example answer pattern:

```text
For this kernel, my hypothesis is memory inefficiency rather than arithmetic cost.
I validate correctness against a CPU reference, then profile with Nsight Compute.
The key evidence is lower useful bandwidth and inefficient memory transactions.
The optimization is to change access pattern/layout/tiling so adjacent lanes access
adjacent memory. I also mention that some metrics such as L2 hit rate can mislead
on streaming kernels because low reuse is expected.
```

---

### 14. End-to-end AI pipeline mapping to this lab

```text
Real AI / camera / RAG pipeline stage             Lab concept that trains it
--------------------------------------------------------------------------------
CPU image preprocessing                            CPU baseline
Tensor normalization / bias / activation           vector add
Non-contiguous tensor slice                        strided access
Fused custom plugin kernel                         occupancy/register pressure
Mean/variance/score aggregation                    reduction
NHWC <-> NCHW layout conversion                    transpose
Tiled image filter / convolution                    shared-memory bank conflict
Detection metadata / tracking records              AoS vs SoA
Camera frame upload / tensor upload                 pageable vs pinned copy
Profiler-based optimization report                  ncu workflow
```

This is the discussion bridge:

```text
The lab is not isolated from real AI work. Each microbenchmark represents a
failure mode that appears in production AI pipelines: copy overhead, layout
mismatch, poor coalescing, register pressure, atomics, synchronization, and
misleading profiler metrics.
```

---

### 15. One-page whiteboard summary

```text
                           GPU Optimization Whiteboard

Input data
   │
   ├── CPU preprocessing bottleneck?
   │       └── measure CPU scalar/OpenMP baseline
   │
   ├── Host-device copy bottleneck?
   │       └── compare pageable vs pinned copy
   │
   ▼
GPU kernel
   │
   ├── Memory-bound?
   │       ├── coalesced vector add
   │       ├── strided access
   │       └── AoS vs SoA
   │
   ├── Cache/reuse/layout issue?
   │       └── transpose tiled shared-memory path
   │
   ├── Register/occupancy issue?
   │       └── low vs high register kernel
   │
   ├── Atomic/synchronization issue?
   │       └── reduction atomic vs staged reduction
   │
   └── Shared-memory conflict?
           └── 32x32 vs 32x33 tile padding

Validation
   │
   └── CPU reference / PASS / FAIL

Profiling
   │
   └── Nsight Compute:
       kernel time, SM throughput, L2/DRAM, occupancy,
       register count, spills, transactions, bank conflicts, stall reasons

Decision
   │
   └── optimize only after the bottleneck is classified
```

---



## Example scenarios and how to explain them

This section is now expanded with **scenario-specific text-view diagrams**. Each scenario has four parts:

```text
1. Real-world use case
2. Code path in this lab
3. Data-flow / hardware-flow diagram
4. What to validate and what to profile
```

Use these diagrams during  discussion as a whiteboard replacement. They help you move from code syntax to system-level explanation.

---

### 1. CPU baseline

Command:

```bash
./build/gpu_cpu_optimization cpu 16777216
```

Scenario: AI preprocessing on CPU, such as image normalization, token preprocessing, feature packing, or tensor preparation.

 explanation: First establish CPU baseline. If CPU memory bandwidth or preprocessing dominates, optimizing only GPU kernel time will not fix end-to-end latency.

Validation: CPU result is deterministic; this section mainly establishes timing.

#### Text-view diagram: CPU scalar vs CPU parallel baseline

```text
Real AI pipeline case
─────────────────────
Camera frame / token batch / feature buffer
        │
        ▼
CPU preprocessing
  - normalize pixels
  - convert layout
  - pack tensors
  - prepare metadata
        │
        ▼
GPU upload / inference
```

```text
Lab code path
─────────────
main()
  │
  ├── command == "cpu" or "all"
  │
  ├── allocate vectors:
  │      h_a[N], h_b[N], h_c[N]
  │
  ├── fill_vector(h_a)
  ├── fill_vector(h_b)
  │
  ├── cpu_vector_add_scalar()
  │      for i = 0 .. N-1:
  │          c[i] = a[i] + b[i]
  │
  └── cpu_vector_add_parallel()
         OpenMP splits index range across CPU cores
```

```text
CPU scalar memory path
──────────────────────
One CPU core
   │
   ├── load a[i]
   ├── load b[i]
   ├── floating-point add
   └── store c[i]

DRAM/cache traffic per element:
   a[i] load  = 4 bytes
   b[i] load  = 4 bytes
   c[i] store = 4 bytes
   total useful traffic ≈ 12 bytes per element
```

```text
CPU OpenMP memory path
──────────────────────
              Shared memory controller / DRAM bandwidth
                              ▲
                              │
       ┌──────────────┬───────┴───────┬──────────────┐
       │              │               │              │
       ▼              ▼               ▼              ▼
    CPU core 0     CPU core 1      CPU core 2     CPU core 3 ...
    chunk 0        chunk 1         chunk 2        chunk 3

Expected behavior:
  More CPU cores can increase throughput until shared memory bandwidth saturates.
  After saturation, more threads may give little gain or even add overhead.
```

```text
Discussion whiteboard
─────────────────────
Question: Why measure CPU baseline for a GPU ?

Answer:
  Because real AI latency is not only CUDA kernel time.
  If CPU preprocessing is memory-bandwidth-bound, then optimizing the GPU kernel
  will not solve end-to-end latency.
```

What to profile or observe:

```text
Linux tools:
  perf stat ./build/gpu_cpu_optimization cpu 16777216
  htop
  numactl --hardware

Benchmark signals:
  cpu_vector_add_scalar time_ms
  cpu_vector_add_parallel time_ms
  scaling from scalar to parallel

 conclusion:
  CPU parallelism helps only until memory bandwidth or scheduling overhead dominates.
```

---

### 2. GPU coalesced vector add

Command:

```bash
./build/gpu_cpu_optimization vector 16777216
```

Scenario: elementwise tensor operations such as bias add, activation, normalization, post-processing, or simple image transform.

 explanation: This is memory-bandwidth-bound. The kernel performs two loads and one store for one add. L2 hit rate can be misleading because there is little temporal reuse.

Profile:

```bash
ncu --set full --target-processes all --force-overwrite \
    -o reports/ncu_vector \
    ./build/gpu_cpu_optimization vector 16777216 \
    > reports/ncu_vector.txt 2>&1
```

Look at: DRAM throughput, L2 throughput, memory workload analysis, achieved occupancy, warp stall reasons.

#### Text-view diagram: coalesced vector-add execution

```text
Real AI pipeline case
─────────────────────
Tensor operation after model layer
  - bias add
  - activation
  - normalization
  - confidence score scaling
  - image pixel transform

Pattern:
  output[i] = input_a[i] + input_b[i]
```

```text
Lab code path
─────────────
run_gpu_vector_add(n, iters, strided=false)
  │
  ├── allocate host vectors
  ├── compute CPU reference h_ref[i] = h_a[i] + h_b[i]
  │
  ├── DeviceBuffer<float> d_a, d_b, d_c
  ├── cudaMemcpy h_a -> d_a
  ├── cudaMemcpy h_b -> d_b
  │
  ├── launch vector_add_coalesced_kernel<<<grid, block>>>()
  │
  ├── time repeated kernel launches with GpuTimer
  │
  ├── cudaMemcpy d_c -> h_c
  │
  └── approx_equal(h_c, h_ref)
```

```text
CUDA Internals flow:
───────────────────

CPU host code
  -> CUDA grid
     -> thread blocks
        -> SMs
           -> warps of 32 threads
              -> lanes
                 -> memory transactions / shared memory / registers

CUDA execution view
───────────────────
Grid
 ├── Block 0
 │    ├── Warp 0: lanes 0..31
 │    ├── Warp 1: lanes 0..31
 │    └── ...
 ├── Block 1
 └── ...

Thread index:
  i = blockIdx.x * blockDim.x + threadIdx.x

Grid-stride loop:
  i, i + gridDim.x * blockDim.x, i + 2*stride, ...
```

```text
Coalesced memory access for one warp
────────────────────────────────────
Warp lanes:    0      1      2      3              31
Index i:       i      i+1    i+2    i+3      ...   i+31

Load A:      A[i]   A[i+1] A[i+2] A[i+3]    ...   A[i+31]
Load B:      B[i]   B[i+1] B[i+2] B[i+3]    ...   B[i+31]
Store C:     C[i]   C[i+1] C[i+2] C[i+3]    ...   C[i+31]

Memory layout:
A: ┌────┬────┬────┬────┬────┬────┬────┐
   │A[i]│A+1 │A+2 │A+3 │... │A+30│A+31│
   └────┴────┴────┴────┴────┴────┴────┘
          contiguous addresses
```

```text
Performance model
─────────────────
For each element:
  2 float loads + 1 float store + 1 add

Useful bytes:
  4 + 4 + 4 = 12 bytes per element

Arithmetic:
  1 floating-point add per element

Conclusion:
  Very low arithmetic intensity
        │
        ▼
  Usually memory-bandwidth-bound, not compute-bound
```

```text
Profiler interpretation
───────────────────────
If performance is good:
  - high DRAM/L2 throughput
  - efficient global load/store pattern
  - low instruction complexity

Metric that can mislead:
  L2 hit rate

Why:
  Data is streamed once. Low reuse means low L2 hit rate may be normal.
  Do not optimize just to increase L2 hit rate if achieved bandwidth is already high.
```

 conclusion:

```text
This benchmark proves I understand the difference between arithmetic work and
memory traffic. Vector add is simple, but it is a clean test of bandwidth,
coalescing, timing, validation, and profiler interpretation.
```

---

### 3. Strided/uncoalesced access

Command:

```bash
./build/gpu_cpu_optimization stride 16777216
```

Scenario: tensor slicing, wrong image layout, AoS metadata access, gather/scatter patterns, or non-contiguous frame/tensor memory.

 explanation: Same math as vector add, but adjacent warp lanes access far-apart addresses. This increases memory transactions and lowers useful bandwidth.

Look at: global memory transactions, sector/request ratio, memory stalls, useful GB/s.

#### Text-view diagram: strided access and coalescing failure

```text
Real AI pipeline case
─────────────────────
Examples:
  - reading every 8th tensor element
  - channel extraction from interleaved image data
  - non-contiguous tensor slice
  - bad layout conversion
  - accessing one field inside an array of large records
```

```text
Lab code path
─────────────
run_gpu_vector_add(n, iters, strided=true)
  │
  ├── stride = 8
  ├── logical_n = n / stride
  │
  └── vector_add_strided_kernel<<<grid, block>>>()
         i = global thread index
         j = i * stride
         c[j] = a[j] + b[j]
```

```text
Coalesced case for comparison
─────────────────────────────
Lane:       0    1    2    3    4          31
Address:   A0   A1   A2   A3   A4   ...   A31

Result:
  Adjacent lanes -> adjacent addresses -> fewer memory transactions
```

```text
Strided case with stride = 8
────────────────────────────
Lane:       0    1    2    3    4              31
Logical i:  0    1    2    3    4       ...    31
Address:   A0   A8   A16  A24  A32      ...    A248

Memory picture:
A0 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 ... A248
│                       │                 │
└─ lane 0               └─ lane 1         └─ lane 31 far away
```

```text
Hardware consequence
────────────────────
Warp issues memory instruction
        │
        ▼
Addresses are spread across memory sectors/cache lines
        │
        ▼
More memory transactions are needed
        │
        ▼
Many fetched bytes are not useful for this warp
        │
        ▼
Effective useful GB/s drops
```

```text
Profiler decision map
─────────────────────
Observation in profiler                 Interpretation
──────────────────────────────────────  ──────────────────────────────
Low useful GB/s                          poor memory efficiency
High sectors/request                     uncoalesced access
High memory dependency stalls            waiting on memory
Similar instruction count to vector add  math is not the reason
```

 conclusion:

```text
This is the same arithmetic as vector add, but slower because of address pattern.
That is the key GPU lesson: memory layout can dominate arithmetic cost.
```

---

### 4. Occupancy/register pressure

Command:

```bash
./build/gpu_cpu_optimization occupancy 16777216
```

Scenario: fused kernels, long expressions, shader-like code, or TensorRT/custom plugin kernels with many live temporaries.

 explanation: Occupancy is useful only if the kernel is latency-limited. More occupancy is not always better. Register pressure can reduce occupancy or cause local-memory spills.

Look at: ptxas register count, local memory/spills, achieved occupancy, eligible warps per scheduler, stall reasons.

#### Text-view diagram: occupancy is a resource-balance problem

```text
Real AI pipeline case
─────────────────────
A fused custom kernel may do many operations in one pass:
  - load tensor value
  - apply scale
  - apply bias
  - activation
  - clamp
  - write result
  - update statistics

Benefit:
  fewer memory passes

Risk:
  many live variables -> high register pressure
```

```text
Lab code path
─────────────
run_occupancy(n, iters, high=false)
  └── low_register_kernel()

run_occupancy(n, iters, high=true)
  └── high_register_kernel()
       r0, r1, r2, ... r15 stay live
       repeated math keeps many registers active
```

```text
SM resource model
─────────────────
One SM has finite resources:

┌───────────────────────────────────────────────┐
│ Streaming Multiprocessor / SM                 │
│                                               │
│ Register file      ──┐                        │
│ Shared memory      ──┼── limits how many      │
│ Warp slots         ──┤   blocks/warps can     │
│ Block slots        ──┘   reside at once       │
│                                               │
│ Warp schedulers issue ready warp instructions │
└───────────────────────────────────────────────┘
```

```text
Low-register kernel
───────────────────
Few live variables per thread
        │
        ▼
Lower registers/thread
        │
        ▼
More warps can fit on SM
        │
        ▼
Potentially higher occupancy
```

```text
High-register kernel
────────────────────
Many live variables per thread
        │
        ▼
Higher registers/thread
        │
        ▼
Fewer warps can fit on SM
        │
        ▼
Lower occupancy possible
        │
        ├── if still enough ready warps: performance may be OK
        │
        └── if not enough ready warps: latency hiding suffers
```

```text
Register spill path
───────────────────
Too many live variables
        │
        ▼
Compiler cannot keep all values in registers
        │
        ▼
Some values spill to local memory
        │
        ▼
Local memory is backed by global memory path
        │
        ▼
Kernel becomes slower
```

```text
 caveat
────────────────
Do not say:
  "Higher occupancy is always better."

Say:
  "Occupancy is a latency-hiding signal. I check it together with eligible warps,
   stall reasons, register count, spills, and memory throughput."
```

Validation and profiling:

```text
Build with ptxas report:
  nvcc ... -Xptxas=-v ...

Check:
  registers per thread
  spill stores / spill loads
  achieved occupancy
  eligible warps per scheduler
  execution dependency stalls
  memory dependency stalls
```

---

### 5. Reduction

Command:

```bash
./build/gpu_cpu_optimization reduction 16777216
```

Scenario: sum, mean, variance, histogram-like aggregation, normalization statistics, metric aggregation, or detection filtering.

 explanation: Avoid one global atomic per thread/block when possible. Reduce locally with warp shuffle and shared memory, then write one partial result per block.

Look at: atomic stalls, synchronization stalls, memory throughput, kernel duration.

#### Text-view diagram: atomic contention vs hierarchical reduction

```text
Real AI pipeline case
─────────────────────
Reduction appears in:
  - sum of tensor values
  - mean/variance for normalization
  - confidence-score aggregation
  - histogram or bin accumulation
  - metric collection from many detections
```

```text
Lab code path
─────────────
run_reduction(n, iters, atomic_bad=true)
  └── reduce_atomic_bad_kernel()
       each thread accumulates local sum
       atomicAdd(out, sum)

run_reduction(n, iters, atomic_bad=false)
  └── reduce_block_warp_kernel()
       local per-thread sum
       warp_reduce_sum()
       shared warp_sums[]
       first warp reduces warp sums
       partial[blockIdx.x] = block_sum
```

```text
Bad global atomic design
────────────────────────
Many threads/warps/blocks
        │
        ▼
all update one global memory location
        │
        ▼
atomicAdd(out, sum)
        │
        ▼
serialization / contention
        │
        ▼
poor scaling
```

```text
Bad atomic diagram
──────────────────
Thread 0  ─┐
Thread 1  ─┤
Thread 2  ─┤
Thread 3  ─┤
...        ├── atomicAdd(global_out)
Thread N  ─┘

Single hot global address becomes the bottleneck.
```

```text
Optimized hierarchical reduction
────────────────────────────────
Input array
   │
   ▼
Each thread accumulates several elements in a register
   │
   ▼
Warp-level reduction using __shfl_down_sync()
   │
   ▼
One value per warp written to shared memory
   │
   ▼
First warp reduces warp_sums[]
   │
   ▼
One partial sum per block written to global memory
```

```text
Block-level view
────────────────
┌────────────────────────── Block k ──────────────────────────┐
│                                                             │
│ Warp 0: lane sums -> shuffle reduce -> warp_sum_0            │
│ Warp 1: lane sums -> shuffle reduce -> warp_sum_1            │
│ Warp 2: lane sums -> shuffle reduce -> warp_sum_2            │
│ ...                                                         │
│                                                             │
│ shared memory: warp_sums[0..num_warps-1]                    │
│                                                             │
│ First warp: reduce warp_sums[] -> block_sum                 │
│                                                             │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
                         partial[k]
```

```text
Profiler interpretation
───────────────────────
Bad version:
  high atomic stalls, serialization, poor scaling

Optimized version:
  fewer global updates, more work in registers/shared memory,
  possible synchronization cost, usually better scalability
```

 conclusion:

```text
Reduction optimization is about reducing globally visible contention. Do as much
work locally as possible, then write fewer results to global memory.
```

---

### 6. Matrix transpose

Command:

```bash
./build/gpu_cpu_optimization transpose
```

Scenario: NHWC↔NCHW conversion, matrix preparation for GEMM, attention cache layout transformation, image/tensor format conversion.

 explanation: Naive transpose is usually uncoalesced on either read or write. Shared-memory tiling improves global coalescing. Padding the tile reduces shared-memory bank conflicts.

Look at: global memory efficiency, shared-memory bank conflicts, L2/DRAM throughput.

#### Text-view diagram: naive transpose vs tiled transpose

```text
Real AI pipeline case
─────────────────────
Common layout transforms:
  NHWC -> NCHW
  NCHW -> NHWC
  matrix transpose before GEMM
  attention cache layout transform
  image/tensor format conversion
```

```text
Lab code path
─────────────
run_transpose()
  │
  ├── transpose_naive_kernel()
  │      direct global read and direct global write
  │
  └── transpose_tiled_kernel()
         global -> shared tile[32][33] -> global
```

```text
Naive transpose memory problem
──────────────────────────────
Input A is row-major:

A row 0: A00 A01 A02 A03 ... contiguous
A row 1: A10 A11 A12 A13 ... contiguous
A row 2: A20 A21 A22 A23 ... contiguous

Transpose writes:
B[col][row] = A[row][col]

A warp may read adjacent A values, but write B values far apart.
```

```text
Naive access picture
────────────────────
Read from A:
  lane 0 -> A[0][0]
  lane 1 -> A[0][1]
  lane 2 -> A[0][2]
  lane 3 -> A[0][3]
  coalesced read

Write to B:
  lane 0 -> B[0][0]
  lane 1 -> B[1][0]
  lane 2 -> B[2][0]
  lane 3 -> B[3][0]
  strided write in row-major layout
```

```text
Tiled transpose design
──────────────────────
Step 1: load tile from global A into shared memory
        coalesced global reads

Step 2: synchronize block
        __syncthreads()

Step 3: write transposed tile from shared memory to global B
        coalesced global writes
```

```text
Tiled flow
──────────
Global A tile
┌─────────────────────┐
│ contiguous row data  │
└──────────┬──────────┘
           │ coalesced read
           ▼
Shared memory tile[32][33]
┌─────────────────────┐
│ 32 rows              │
│ 33 columns           │  +1 padding changes bank mapping
└──────────┬──────────┘
           │ transposed read from shared memory
           ▼
Global B tile
┌─────────────────────┐
│ contiguous row write │
└─────────────────────┘
```

```text
Why padding matters
───────────────────
tile[32][32]
  column read may map many lanes to same bank

 tile[32][33]
  each row starts one float later than a power-of-32 boundary
  bank mapping spreads across banks
```

 conclusion:

```text
Transpose is a strong example because it combines global coalescing,
shared-memory tiling, synchronization, and bank-conflict avoidance.
```

---

### 7. Shared-memory bank conflict

Command:

```bash
./build/gpu_cpu_optimization bank
```

Scenario: tiled convolution, stencil filters, transpose, local image filters, and any shared-memory column read.

 explanation: Shared memory is banked. If many lanes access different addresses mapping to the same bank, accesses serialize. Padding changes bank mapping.

Look at: shared-memory conflict metrics and warp stall reasons.

#### Text-view diagram: shared-memory bank conflict and padding

```text
Real AI pipeline case
─────────────────────
Shared memory is used in:
  - tiled convolution
  - stencil filters
  - local image filters
  - Sobel/blur/sharpen filters
  - transpose
  - block-level reductions
```

```text
Lab code path
─────────────
run_bank_conflict()
  │
  ├── shared_bank_conflict_kernel()
  │      __shared__ float tile[32][32]
  │      column-style access can conflict
  │
  └── shared_bank_padded_kernel()
         __shared__ float tile[32][33]
         padding changes bank mapping
```

```text
Shared memory bank model
────────────────────────
Simplified 32-bank model:

Bank:       0    1    2    3    4          31
Address:    0    1    2    3    4    ...   31
Address:   32   33   34   35   36    ...   63
Address:   64   65   66   67   68    ...   95
```

```text
Conflict case: tile[32][32]
───────────────────────────
Column access by one warp:

lane 0  -> tile[0][0]   -> bank 0
lane 1  -> tile[1][0]   -> bank 0
lane 2  -> tile[2][0]   -> bank 0
...
lane31  -> tile[31][0]  -> bank 0

Many lanes target same bank
        │
        ▼
accesses serialize
        │
        ▼
shared memory becomes slower than expected
```

```text
Padded case: tile[32][33]
─────────────────────────
Column access by one warp:

lane 0  -> tile[0][0]   -> bank 0
lane 1  -> tile[1][0]   -> bank 1
lane 2  -> tile[2][0]   -> bank 2
...
lane31  -> tile[31][0]  -> bank 31

Many lanes target different banks
        │
        ▼
parallel bank service
        │
        ▼
less serialization
```

```text
 whiteboard statement
──────────────────────────────
Padding does not change the algorithm or output. It changes address-to-bank
mapping so the same logical operation better matches the hardware.
```

Profiler interpretation:

```text
Look for:
  shared load/store bank conflicts
  warp stalls related to shared memory
  runtime difference between 32x32 and 32x33 tile
```

---

### 8. AoS vs SoA

Command:

```bash
./build/gpu_cpu_optimization layout 16777216
```

Scenario: bounding-box metadata, particle/state arrays, detection results, tracking records, or camera frame metadata.

 explanation: AoS is convenient for CPU object code. SoA is often better for GPU field-wise access because each field is contiguous, improving coalescing.

Look at: useful GB/s, memory transactions, L2/DRAM throughput.

#### Text-view diagram: data layout as an optimization

```text
Real AI pipeline case
─────────────────────
Examples:
  - object detection boxes
  - tracking records
  - particle/state arrays
  - camera frame metadata
  - post-processing records with x, y, w, h, score, class_id
```

```text
Lab code path
─────────────
run_aos_soa(n, iters, aos=true)
  └── update_aos_kernel()
       ParticleAoS { x, y, z, vx, vy, vz }

run_aos_soa(n, iters, aos=false)
  └── update_soa_kernel()
       x[], y[], z[], vx[], vy[], vz[]
```

```text
AoS: Array of Structures
────────────────────────
struct ParticleAoS {
    float x, y, z, vx, vy, vz;
};

Memory:
P0: x y z vx vy vz | P1: x y z vx vy vz | P2: x y z vx vy vz | ...
```

```text
AoS field-wise GPU access
─────────────────────────
Kernel updates only x using vx:
  p[i].x += p[i].vx * dt

Warp lanes:
lane 0 -> P0.x and P0.vx
lane 1 -> P1.x and P1.vx
lane 2 -> P2.x and P2.vx

Memory addresses for x:
P0.x           P1.x           P2.x           P3.x
 │              │              │              │
 ▼              ▼              ▼              ▼
separated by sizeof(ParticleAoS), not contiguous floats
```

```text
SoA: Structure of Arrays
────────────────────────
float x[N], y[N], z[N], vx[N], vy[N], vz[N];

Memory:
x:  x0  x1  x2  x3  x4  ... contiguous
vx: vx0 vx1 vx2 vx3 vx4 ... contiguous
```

```text
SoA field-wise GPU access
─────────────────────────
Kernel updates:
  x[i] += vx[i] * dt

Warp lanes:
lane 0 -> x[0], vx[0]
lane 1 -> x[1], vx[1]
lane 2 -> x[2], vx[2]

Adjacent lanes access adjacent memory
        │
        ▼
better coalescing and better useful bandwidth
```

```text
Layout decision tree
────────────────────
Does each thread use the full object?
        │
        ├── yes
        │     AoS can be acceptable and simpler
        │
        └── no, each kernel uses only selected fields
              │
              ▼
            SoA is usually better for GPU coalescing
```

 conclusion:

```text
Sometimes the best kernel optimization is not changing arithmetic. It is changing
how data is laid out so that the hardware can fetch useful data efficiently.
```

---

### 9. Pageable vs pinned host-device copy

Command:

```bash
./build/gpu_cpu_optimization copy 16777216
```

Scenario: camera frame upload to GPU, CPU-generated tensor upload, GPU output download, streaming inference.

 explanation: End-to-end AI latency may be dominated outside the kernel. Pinned memory is page-locked and more suitable for DMA transfers.

Look at: H2D copy time, end-to-end latency, Nsight Systems timeline if extending the lab.

#### Text-view diagram: host-device copy as an end-to-end bottleneck

```text
Real AI pipeline case
─────────────────────
Camera / network / disk / CPU preprocessing
        │
        ▼
CPU memory frame or tensor
        │
        ▼
Host-to-device copy
        │
        ▼
GPU kernel / inference
        │
        ▼
Device-to-host copy or next GPU stage
        │
        ▼
post-processing / storage / response
```

```text
Lab code path
─────────────
run_copy(n, iters, pinned=false)
  └── std::vector<float> pageable host memory
      cudaMemcpy HostToDevice

run_copy(n, iters, pinned=true)
  └── PinnedHostBuffer<float>
      cudaMallocHost page-locked memory
      cudaMemcpy HostToDevice
```

```text
Pageable host memory path
─────────────────────────
Application std::vector memory
        │
        │ cudaMemcpy HostToDevice
        ▼
CUDA driver may stage through temporary pinned memory
        │
        ▼
DMA over PCIe
        │
        ▼
GPU global memory
```

```text
Pinned host memory path
───────────────────────
Application pinned/page-locked memory
        │
        │ cudaMemcpy HostToDevice
        ▼
DMA over PCIe
        │
        ▼
GPU global memory
```

```text
End-to-end latency lesson
────────────────────────
Total pipeline time:

T_total = T_cpu_preprocess
        + T_host_to_device_copy
        + T_kernel
        + T_device_to_host_copy
        + T_postprocess

If T_host_to_device_copy is large:
  optimizing T_kernel alone may show no visible end-to-end improvement.
```

```text
Profiler/tooling view
─────────────────────
Nsight Compute:
  good for kernel-level details

Nsight Systems:
  better for timeline view:
    CPU work | cudaMemcpy | kernel launch | GPU execution | synchronization

Linux timing:
  useful for end-to-end wall-clock behavior
```

 conclusion:

```text
I separate kernel optimization from pipeline optimization. For streaming AI,
copy overhead, pinned memory, batching, and overlap can matter as much as CUDA
kernel speed.
```

---
##  reporting template

```text
Case:
Input size:
Baseline result:
Optimized result:
Correctness:
Speedup:
Profiler evidence:
Main bottleneck:
Metric that mattered:
Metric that could mislead:
Code change:
Why it helped:
```

---


# Line-by-line explanation: `cuda_check.hpp`

| Line | Section | Code | Explanation |
|---:|---|---|---|
| 0001 | header guard and includes | `#pragma once` | Prevents multiple inclusion of this header in the same translation unit. |
| 0002 | header guard and includes | &nbsp; | Blank line separating the header guard and includes code for readability. |
| 0003 | header guard and includes | `#include <cuda_runtime.h>` | Includes a dependency needed by the header guard and includes code. |
| 0004 | header guard and includes | `#include <cstdio>` | Includes a dependency needed by the header guard and includes code. |
| 0005 | header guard and includes | `#include <cstdlib>` | Includes a dependency needed by the header guard and includes code. |
| 0006 | header guard and includes | `#include <stdexcept>` | Includes a dependency needed by the header guard and includes code. |
| 0007 | header guard and includes | `#include <string>` | Includes a dependency needed by the header guard and includes code. |
| 0008 | general header code | &nbsp; | Blank line separating the general header code code for readability. |
| 0009 | CUDA error-checking macros | `#define CUDA_CHECK(call) do { \` | Defines the CUDA runtime error-checking macro used throughout the lab. |
| 0010 | CUDA error-checking macros | `    cudaError_t err__ = (call); \` | Captures the return status of a CUDA runtime call. |
| 0011 | CUDA error-checking macros | `    if (err__ != cudaSuccess) { \` | Checks whether the CUDA operation succeeded. |
| 0012 | CUDA error-checking macros | `        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \` | Converts CUDA error code into readable text. |
| 0013 | CUDA error-checking macros | `        std::exit(EXIT_FAILURE); \` | Stops execution immediately on CUDA failure so invalid benchmark data is not trusted. |
| 0014 | CUDA error-checking macros | `    } \` | C++/CUDA statement within the CUDA error-checking macros section. |
| 0015 | CUDA error-checking macros | `} while (0)` | C++/CUDA statement within the CUDA error-checking macros section. |
| 0016 | CUDA error-checking macros | &nbsp; | Blank line separating the CUDA error-checking macros code for readability. |
| 0017 | CUDA error-checking macros | `#define CUDA_KERNEL_CHECK() do { \` | Defines the kernel launch/execution error-checking macro. |
| 0018 | CUDA error-checking macros | `    CUDA_CHECK(cudaGetLastError()); \` | C++/CUDA statement within the CUDA error-checking macros section. |
| 0019 | CUDA error-checking macros | `    CUDA_CHECK(cudaDeviceSynchronize()); \` | C++/CUDA statement within the CUDA error-checking macros section. |
| 0020 | CUDA error-checking macros | `} while (0)` | C++/CUDA statement within the CUDA error-checking macros section. |
| 0021 | general header code | &nbsp; | Blank line separating the general header code code for readability. |
| 0022 | GpuTimer RAII CUDA event timer | `class GpuTimer {` | Declares RAII timer class built on CUDA events. |
| 0023 | GpuTimer RAII CUDA event timer | `public:` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0024 | GpuTimer RAII CUDA event timer | `    GpuTimer() {` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0025 | GpuTimer RAII CUDA event timer | `        CUDA_CHECK(cudaEventCreate(&start_));` | Allocates a CUDA event used for GPU-side timing. |
| 0026 | GpuTimer RAII CUDA event timer | `        CUDA_CHECK(cudaEventCreate(&stop_));` | Allocates a CUDA event used for GPU-side timing. |
| 0027 | GpuTimer RAII CUDA event timer | `    }` | Scope delimiter for the GpuTimer RAII CUDA event timer code. |
| 0028 | GpuTimer RAII CUDA event timer | `    ~GpuTimer() {` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0029 | GpuTimer RAII CUDA event timer | `        cudaEventDestroy(start_);` | Releases a CUDA event resource. |
| 0030 | GpuTimer RAII CUDA event timer | `        cudaEventDestroy(stop_);` | Releases a CUDA event resource. |
| 0031 | GpuTimer RAII CUDA event timer | `    }` | Scope delimiter for the GpuTimer RAII CUDA event timer code. |
| 0032 | GpuTimer RAII CUDA event timer | `    GpuTimer(const GpuTimer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0033 | GpuTimer RAII CUDA event timer | `    GpuTimer& operator=(const GpuTimer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0034 | GpuTimer RAII CUDA event timer | &nbsp; | Blank line separating the GpuTimer RAII CUDA event timer code for readability. |
| 0035 | GpuTimer RAII CUDA event timer | `    void start(cudaStream_t stream = 0) { CUDA_CHECK(cudaEventRecord(start_, stream)); }` | Records a CUDA event in a stream to mark a timestamp. |
| 0036 | GpuTimer RAII CUDA event timer | `    float stop_ms(cudaStream_t stream = 0) {` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0037 | GpuTimer RAII CUDA event timer | `        CUDA_CHECK(cudaEventRecord(stop_, stream));` | Records a CUDA event in a stream to mark a timestamp. |
| 0038 | GpuTimer RAII CUDA event timer | `        CUDA_CHECK(cudaEventSynchronize(stop_));` | Waits until the stop event completes, making timing valid. |
| 0039 | GpuTimer RAII CUDA event timer | `        float ms = 0.0f;` | Stores average runtime in milliseconds. |
| 0040 | GpuTimer RAII CUDA event timer | `        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));` | Calculates GPU elapsed time between two CUDA events. |
| 0041 | GpuTimer RAII CUDA event timer | `        return ms;` | Returns from the current function in the GpuTimer RAII CUDA event timer section. |
| 0042 | GpuTimer RAII CUDA event timer | `    }` | Scope delimiter for the GpuTimer RAII CUDA event timer code. |
| 0043 | GpuTimer RAII CUDA event timer | `private:` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0044 | GpuTimer RAII CUDA event timer | `    cudaEvent_t start_{};` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0045 | GpuTimer RAII CUDA event timer | `    cudaEvent_t stop_{};` | C++/CUDA statement within the GpuTimer RAII CUDA event timer section. |
| 0046 | GpuTimer RAII CUDA event timer | `};` | Scope delimiter for the GpuTimer RAII CUDA event timer code. |
| 0047 | general header code | &nbsp; | Blank line separating the general header code code for readability. |
| 0048 | DeviceBuffer RAII device-memory wrapper | `template <typename T>` | C++/CUDA statement within the DeviceBuffer RAII device-memory wrapper section. |
| 0049 | DeviceBuffer RAII device-memory wrapper | `class DeviceBuffer {` | Declares move-only RAII wrapper for GPU memory. |
| 0050 | DeviceBuffer RAII device-memory wrapper | `public:` | C++/CUDA statement within the DeviceBuffer RAII device-memory wrapper section. |
| 0051 | DeviceBuffer RAII device-memory wrapper | `    explicit DeviceBuffer(size_t count) : count_(count) {` | Tracks the number of elements in the managed buffer. |
| 0052 | DeviceBuffer RAII device-memory wrapper | `        CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));` | Allocates GPU global memory. |
| 0053 | DeviceBuffer RAII device-memory wrapper | `    }` | Scope delimiter for the DeviceBuffer RAII device-memory wrapper code. |
| 0054 | DeviceBuffer RAII device-memory wrapper | `    ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }` | Frees GPU global memory. |
| 0055 | DeviceBuffer RAII device-memory wrapper | `    DeviceBuffer(const DeviceBuffer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0056 | DeviceBuffer RAII device-memory wrapper | `    DeviceBuffer& operator=(const DeviceBuffer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0057 | DeviceBuffer RAII device-memory wrapper | `    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {` | Implements move semantics so buffer ownership can be transferred safely. |
| 0058 | DeviceBuffer RAII device-memory wrapper | `        other.ptr_ = nullptr;` | Initializes or clears a pointer to prevent invalid/free-after-move behavior. |
| 0059 | DeviceBuffer RAII device-memory wrapper | `        other.count_ = 0;` | Tracks the number of elements in the managed buffer. |
| 0060 | DeviceBuffer RAII device-memory wrapper | `    }` | Scope delimiter for the DeviceBuffer RAII device-memory wrapper code. |
| 0061 | DeviceBuffer RAII device-memory wrapper | `    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {` | Implements move semantics so buffer ownership can be transferred safely. |
| 0062 | DeviceBuffer RAII device-memory wrapper | `        if (this != &other) {` | Conditional branch used by the DeviceBuffer RAII device-memory wrapper logic. |
| 0063 | DeviceBuffer RAII device-memory wrapper | `            if (ptr_) cudaFree(ptr_);` | Frees GPU global memory. |
| 0064 | DeviceBuffer RAII device-memory wrapper | `            ptr_ = other.ptr_;` | C++/CUDA statement within the DeviceBuffer RAII device-memory wrapper section. |
| 0065 | DeviceBuffer RAII device-memory wrapper | `            count_ = other.count_;` | Tracks the number of elements in the managed buffer. |
| 0066 | DeviceBuffer RAII device-memory wrapper | `            other.ptr_ = nullptr;` | Initializes or clears a pointer to prevent invalid/free-after-move behavior. |
| 0067 | DeviceBuffer RAII device-memory wrapper | `            other.count_ = 0;` | Tracks the number of elements in the managed buffer. |
| 0068 | DeviceBuffer RAII device-memory wrapper | `        }` | Scope delimiter for the DeviceBuffer RAII device-memory wrapper code. |
| 0069 | DeviceBuffer RAII device-memory wrapper | `        return *this;` | Returns from the current function in the DeviceBuffer RAII device-memory wrapper section. |
| 0070 | DeviceBuffer RAII device-memory wrapper | `    }` | Scope delimiter for the DeviceBuffer RAII device-memory wrapper code. |
| 0071 | DeviceBuffer RAII device-memory wrapper | `    T* get() const { return ptr_; }` | Exposes raw pointer for CUDA APIs and kernel launches. |
| 0072 | DeviceBuffer RAII device-memory wrapper | `    size_t count() const { return count_; }` | Tracks the number of elements in the managed buffer. |
| 0073 | DeviceBuffer RAII device-memory wrapper | `    size_t bytes() const { return count_ * sizeof(T); }` | Tracks the number of elements in the managed buffer. |
| 0074 | DeviceBuffer RAII device-memory wrapper | `private:` | C++/CUDA statement within the DeviceBuffer RAII device-memory wrapper section. |
| 0075 | DeviceBuffer RAII device-memory wrapper | `    T* ptr_ = nullptr;` | Initializes or clears a pointer to prevent invalid/free-after-move behavior. |
| 0076 | DeviceBuffer RAII device-memory wrapper | `    size_t count_ = 0;` | Tracks the number of elements in the managed buffer. |
| 0077 | DeviceBuffer RAII device-memory wrapper | `};` | Scope delimiter for the DeviceBuffer RAII device-memory wrapper code. |
| 0078 | general header code | &nbsp; | Blank line separating the general header code code for readability. |
| 0079 | PinnedHostBuffer RAII pinned-host-memory wrapper | `template <typename T>` | C++/CUDA statement within the PinnedHostBuffer RAII pinned-host-memory wrapper section. |
| 0080 | PinnedHostBuffer RAII pinned-host-memory wrapper | `class PinnedHostBuffer {` | Declares RAII wrapper for pinned CPU memory. |
| 0081 | PinnedHostBuffer RAII pinned-host-memory wrapper | `public:` | C++/CUDA statement within the PinnedHostBuffer RAII pinned-host-memory wrapper section. |
| 0082 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    explicit PinnedHostBuffer(size_t count) : count_(count) {` | Tracks the number of elements in the managed buffer. |
| 0083 | PinnedHostBuffer RAII pinned-host-memory wrapper | `        CUDA_CHECK(cudaMallocHost(&ptr_, count_ * sizeof(T)));` | Allocates page-locked host memory for faster/more stable DMA transfers. |
| 0084 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    }` | Scope delimiter for the PinnedHostBuffer RAII pinned-host-memory wrapper code. |
| 0085 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    ~PinnedHostBuffer() { if (ptr_) cudaFreeHost(ptr_); }` | Frees pinned host memory. |
| 0086 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    PinnedHostBuffer(const PinnedHostBuffer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0087 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;` | Deletes copy behavior to prevent double ownership/double free of CUDA resources. |
| 0088 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    T* get() const { return ptr_; }` | Exposes raw pointer for CUDA APIs and kernel launches. |
| 0089 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    T& operator[](size_t i) { return ptr_[i]; }` | C++/CUDA statement within the PinnedHostBuffer RAII pinned-host-memory wrapper section. |
| 0090 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    const T& operator[](size_t i) const { return ptr_[i]; }` | C++/CUDA statement within the PinnedHostBuffer RAII pinned-host-memory wrapper section. |
| 0091 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    size_t count() const { return count_; }` | Tracks the number of elements in the managed buffer. |
| 0092 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    size_t bytes() const { return count_ * sizeof(T); }` | Tracks the number of elements in the managed buffer. |
| 0093 | PinnedHostBuffer RAII pinned-host-memory wrapper | `private:` | C++/CUDA statement within the PinnedHostBuffer RAII pinned-host-memory wrapper section. |
| 0094 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    T* ptr_ = nullptr;` | Initializes or clears a pointer to prevent invalid/free-after-move behavior. |
| 0095 | PinnedHostBuffer RAII pinned-host-memory wrapper | `    size_t count_ = 0;` | Tracks the number of elements in the managed buffer. |
| 0096 | PinnedHostBuffer RAII pinned-host-memory wrapper | `};` | Scope delimiter for the PinnedHostBuffer RAII pinned-host-memory wrapper code. |

# Line-by-line explanation: `gpu_cpu_optimization.cu`

| Line | Section | Code | Explanation |
|---:|---|---|---|
| 0001 | includes/build configuration | `#include "cuda_check.hpp"` | Includes a dependency needed by the includes/build configuration code. |
| 0002 | includes/build configuration | &nbsp; | Blank line separating the includes/build configuration code for readability. |
| 0003 | includes/build configuration | `#include <algorithm>` | Includes a dependency needed by the includes/build configuration code. |
| 0004 | includes/build configuration | `#include <chrono>` | Includes a dependency needed by the includes/build configuration code. |
| 0005 | includes/build configuration | `#include <cmath>` | Includes a dependency needed by the includes/build configuration code. |
| 0006 | includes/build configuration | `#include <cstring>` | Includes a dependency needed by the includes/build configuration code. |
| 0007 | includes/build configuration | `#include <iomanip>` | Includes a dependency needed by the includes/build configuration code. |
| 0008 | includes/build configuration | `#include <iostream>` | Includes a dependency needed by the includes/build configuration code. |
| 0009 | includes/build configuration | `#include <numeric>` | Includes a dependency needed by the includes/build configuration code. |
| 0010 | includes/build configuration | `#include <random>` | Includes a dependency needed by the includes/build configuration code. |
| 0011 | includes/build configuration | `#include <string>` | Includes a dependency needed by the includes/build configuration code. |
| 0012 | includes/build configuration | `#include <vector>` | Includes a dependency needed by the includes/build configuration code. |
| 0013 | includes/build configuration | &nbsp; | Blank line separating the includes/build configuration code for readability. |
| 0014 | includes/build configuration | `#ifdef HAS_OPENMP` | Begins conditional compilation so OpenMP-specific code is compiled only when enabled. |
| 0015 | includes/build configuration | `#include <omp.h>` | Includes a dependency needed by the includes/build configuration code. |
| 0016 | includes/build configuration | `#endif` | Ends the conditional compilation block. |
| 0017 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0018 | Result data model | `struct Result {` | Begins a small record used to store benchmark output. |
| 0019 | Result data model | `    std::string name;` | Stores human-readable benchmark name. |
| 0020 | Result data model | `    float ms;` | Stores average runtime in milliseconds. |
| 0021 | Result data model | `    double gbps;` | Stores effective useful bandwidth. |
| 0022 | Result data model | `    double gops;` | Stores effective operation rate. |
| 0023 | Result data model | `    bool ok;` | Stores validation status. |
| 0024 | Result data model | `};` | Scope delimiter for the Result data model code. |
| 0025 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0026 | CPU wall-clock timer | `static double now_ms_cpu() {` | Defines or calls CPU timing helper for CPU-only benchmarks. |
| 0027 | CPU wall-clock timer | `    using clock = std::chrono::high_resolution_clock;` | Uses C++ chrono clocks to measure CPU elapsed time. |
| 0028 | CPU wall-clock timer | `    static const auto t0 = clock::now();` | C++/CUDA statement within the CPU wall-clock timer section. |
| 0029 | CPU wall-clock timer | `    auto t = clock::now();` | C++/CUDA statement within the CPU wall-clock timer section. |
| 0030 | CPU wall-clock timer | `    return std::chrono::duration<double, std::milli>(t - t0).count();` | Uses C++ chrono clocks to measure CPU elapsed time. |
| 0031 | CPU wall-clock timer | `}` | Scope delimiter for the CPU wall-clock timer code. |
| 0032 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0033 | deterministic input generation | `static void fill_vector(std::vector<float>& v) {` | Allocates host-side vector storage for the deterministic input generation benchmark. |
| 0034 | deterministic input generation | `    std::mt19937 rng(1234);` | Creates deterministic random generator for repeatable inputs. |
| 0035 | deterministic input generation | `    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);` | Creates random float distribution used to fill test data. |
| 0036 | deterministic input generation | `    for (auto& x : v) x = dist(rng);` | Fills each vector element with random test data. |
| 0037 | deterministic input generation | `}` | Scope delimiter for the deterministic input generation code. |
| 0038 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0039 | floating-point validation | `static bool approx_equal(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f) {` | Defines or calls float-vector validation helper. |
| 0040 | floating-point validation | `    if (a.size() != b.size()) return false;` | Conditional branch used by the floating-point validation logic. |
| 0041 | floating-point validation | `    for (size_t i = 0; i < a.size(); ++i) {` | Loop used by the floating-point validation code to repeat work or traverse data. |
| 0042 | floating-point validation | `        if (std::fabs(a[i] - b[i]) > tol) {` | Computes absolute error for validation or relative-error checks. |
| 0043 | floating-point validation | `            std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";` | C++/CUDA statement within the floating-point validation section. |
| 0044 | floating-point validation | `            return false;` | Returns from the current function in the floating-point validation section. |
| 0045 | floating-point validation | `        }` | Scope delimiter for the floating-point validation code. |
| 0046 | floating-point validation | `    }` | Scope delimiter for the floating-point validation code. |
| 0047 | floating-point validation | `    return true;` | Returns from the current function in the floating-point validation section. |
| 0048 | floating-point validation | `}` | Scope delimiter for the floating-point validation code. |
| 0049 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0050 | result formatting | `static void print_result(const Result& r) {` | Defines or calls formatted benchmark result printing. |
| 0051 | result formatting | `    std::cout << std::left << std::setw(34) << r.name` | Formats output so benchmark logs are easier to compare and parse. |
| 0052 | result formatting | `              << " time_ms=" << std::setw(10) << std::fixed << std::setprecision(4) << r.ms` | Formats output so benchmark logs are easier to compare and parse. |
| 0053 | result formatting | `              << " GB/s=" << std::setw(10) << std::setprecision(2) << r.gbps` | Formats output so benchmark logs are easier to compare and parse. |
| 0054 | result formatting | `              << " GOP/s=" << std::setw(10) << std::setprecision(2) << r.gops` | Formats output so benchmark logs are easier to compare and parse. |
| 0055 | result formatting | `              << " status=" << (r.ok ? "PASS" : "FAIL") << "\n";` | C++/CUDA statement within the result formatting section. |
| 0056 | result formatting | `}` | Scope delimiter for the result formatting code. |
| 0057 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0058 | CPU scalar/OpenMP memory-bandwidth baseline | `// -------------------------------------------------------------------------------------------------` | Comment marker for the CPU scalar/OpenMP memory-bandwidth baseline section; it documents the /performance lesson. |
| 0059 | CPU scalar/OpenMP memory-bandwidth baseline | `// 1. CPU baseline: scalar and optional OpenMP.  point: CPU memory bandwidth is also a limit.` | Comment marker for the CPU scalar/OpenMP memory-bandwidth baseline section; it documents the /performance lesson. |
| 0060 | CPU scalar/OpenMP memory-bandwidth baseline | `// -------------------------------------------------------------------------------------------------` | Comment marker for the CPU scalar/OpenMP memory-bandwidth baseline section; it documents the /performance lesson. |
| 0061 | CPU scalar/OpenMP memory-bandwidth baseline | `static Result cpu_vector_add_scalar(const std::vector<float>& a, const std::vector<float>& b,` | Allocates host-side vector storage for the CPU scalar/OpenMP memory-bandwidth baseline benchmark. |
| 0062 | CPU scalar/OpenMP memory-bandwidth baseline | `                                    std::vector<float>& c, int iters) {` | Allocates host-side vector storage for the CPU scalar/OpenMP memory-bandwidth baseline benchmark. |
| 0063 | CPU scalar/OpenMP memory-bandwidth baseline | `    double t0 = now_ms_cpu();` | Defines or calls CPU timing helper for CPU-only benchmarks. |
| 0064 | CPU scalar/OpenMP memory-bandwidth baseline | `    for (int it = 0; it < iters; ++it) {` | Loop used by the CPU scalar/OpenMP memory-bandwidth baseline code to repeat work or traverse data. |
| 0065 | CPU scalar/OpenMP memory-bandwidth baseline | `        for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];` | Performs coalesced vector add; adjacent threads access adjacent floats. |
| 0066 | CPU scalar/OpenMP memory-bandwidth baseline | `    }` | Scope delimiter for the CPU scalar/OpenMP memory-bandwidth baseline code. |
| 0067 | CPU scalar/OpenMP memory-bandwidth baseline | `    double t1 = now_ms_cpu();` | Defines or calls CPU timing helper for CPU-only benchmarks. |
| 0068 | CPU scalar/OpenMP memory-bandwidth baseline | `    double ms = (t1 - t0) / iters;` | C++/CUDA statement within the CPU scalar/OpenMP memory-bandwidth baseline section. |
| 0069 | CPU scalar/OpenMP memory-bandwidth baseline | `    double bytes = 3.0 * a.size() * sizeof(float);` | C++/CUDA statement within the CPU scalar/OpenMP memory-bandwidth baseline section. |
| 0070 | CPU scalar/OpenMP memory-bandwidth baseline | `    return {"cpu_vector_add_scalar", static_cast<float>(ms), bytes / (ms * 1.0e6), a.size() / (ms * 1.0e6), true};` | Converts useful bytes and milliseconds into GB/s. |
| 0071 | CPU scalar/OpenMP memory-bandwidth baseline | `}` | Scope delimiter for the CPU scalar/OpenMP memory-bandwidth baseline code. |
| 0072 | CPU scalar/OpenMP memory-bandwidth baseline | &nbsp; | Blank line separating the CPU scalar/OpenMP memory-bandwidth baseline code for readability. |
| 0073 | CPU scalar/OpenMP memory-bandwidth baseline | `static Result cpu_vector_add_parallel(const std::vector<float>& a, const std::vector<float>& b,` | Allocates host-side vector storage for the CPU scalar/OpenMP memory-bandwidth baseline benchmark. |
| 0074 | CPU scalar/OpenMP memory-bandwidth baseline | `                                      std::vector<float>& c, int iters) {` | Allocates host-side vector storage for the CPU scalar/OpenMP memory-bandwidth baseline benchmark. |
| 0075 | CPU scalar/OpenMP memory-bandwidth baseline | `    double t0 = now_ms_cpu();` | Defines or calls CPU timing helper for CPU-only benchmarks. |
| 0076 | CPU scalar/OpenMP memory-bandwidth baseline | `    for (int it = 0; it < iters; ++it) {` | Loop used by the CPU scalar/OpenMP memory-bandwidth baseline code to repeat work or traverse data. |
| 0077 | CPU scalar/OpenMP memory-bandwidth baseline | `        #ifdef HAS_OPENMP` | Begins conditional compilation so OpenMP-specific code is compiled only when enabled. |
| 0078 | CPU scalar/OpenMP memory-bandwidth baseline | `        #pragma omp parallel for schedule(static)` | Parallelizes CPU loop with OpenMP for CPU baseline comparison. |
| 0079 | CPU scalar/OpenMP memory-bandwidth baseline | `        #endif` | Ends the conditional compilation block. |
| 0080 | CPU scalar/OpenMP memory-bandwidth baseline | `        for (long long i = 0; i < static_cast<long long>(a.size()); ++i) c[i] = a[i] + b[i];` | Performs coalesced vector add; adjacent threads access adjacent floats. |
| 0081 | CPU scalar/OpenMP memory-bandwidth baseline | `    }` | Scope delimiter for the CPU scalar/OpenMP memory-bandwidth baseline code. |
| 0082 | CPU scalar/OpenMP memory-bandwidth baseline | `    double t1 = now_ms_cpu();` | Defines or calls CPU timing helper for CPU-only benchmarks. |
| 0083 | CPU scalar/OpenMP memory-bandwidth baseline | `    double ms = (t1 - t0) / iters;` | C++/CUDA statement within the CPU scalar/OpenMP memory-bandwidth baseline section. |
| 0084 | CPU scalar/OpenMP memory-bandwidth baseline | `    double bytes = 3.0 * a.size() * sizeof(float);` | C++/CUDA statement within the CPU scalar/OpenMP memory-bandwidth baseline section. |
| 0085 | CPU scalar/OpenMP memory-bandwidth baseline | `    return {"cpu_vector_add_parallel", static_cast<float>(ms), bytes / (ms * 1.0e6), a.size() / (ms * 1.0e6), true};` | Converts useful bytes and milliseconds into GB/s. |
| 0086 | CPU scalar/OpenMP memory-bandwidth baseline | `}` | Scope delimiter for the CPU scalar/OpenMP memory-bandwidth baseline code. |
| 0087 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0088 | GPU coalesced vector-add kernel | `// -------------------------------------------------------------------------------------------------` | Comment marker for the GPU coalesced vector-add kernel section; it documents the /performance lesson. |
| 0089 | GPU coalesced vector-add kernel | `// 2. GPU vector add: coalesced streaming kernel.  point: memory-bound; L2 hit rate may mislead.` | Comment marker for the GPU coalesced vector-add kernel section; it documents the /performance lesson. |
| 0090 | GPU coalesced vector-add kernel | `// -------------------------------------------------------------------------------------------------` | Comment marker for the GPU coalesced vector-add kernel section; it documents the /performance lesson. |
| 0091 | GPU coalesced vector-add kernel | `__global__ void vector_add_coalesced_kernel(const float* __restrict__ a,` | Declares a CUDA kernel for the GPU coalesced vector-add kernel case; it runs on GPU. |
| 0092 | GPU coalesced vector-add kernel | `                                            const float* __restrict__ b,` | Tells compiler pointers do not alias, which can help optimization. |
| 0093 | GPU coalesced vector-add kernel | `                                            float* __restrict__ c,` | Tells compiler pointers do not alias, which can help optimization. |
| 0094 | GPU coalesced vector-add kernel | `                                            size_t n) {` | C++/CUDA statement within the GPU coalesced vector-add kernel section. |
| 0095 | GPU coalesced vector-add kernel | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0096 | GPU coalesced vector-add kernel | `    size_t stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0097 | GPU coalesced vector-add kernel | `    for (; i < n; i += stride) c[i] = a[i] + b[i];` | Grid-stride loop lets each thread process multiple elements safely. |
| 0098 | GPU coalesced vector-add kernel | `}` | Scope delimiter for the GPU coalesced vector-add kernel code. |
| 0099 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0100 | GPU strided/uncoalesced vector-add kernel | `// -------------------------------------------------------------------------------------------------` | Comment marker for the GPU strided/uncoalesced vector-add kernel section; it documents the /performance lesson. |
| 0101 | GPU strided/uncoalesced vector-add kernel | `// 3. Strided/uncoalesced access: adjacent warp lanes touch far-apart memory.` | Comment marker for the GPU strided/uncoalesced vector-add kernel section; it documents the /performance lesson. |
| 0102 | GPU strided/uncoalesced vector-add kernel | `//  point: high transaction count and poor memory efficiency.` | Comment marker for the GPU strided/uncoalesced vector-add kernel section; it documents the /performance lesson. |
| 0103 | GPU strided/uncoalesced vector-add kernel | `// -------------------------------------------------------------------------------------------------` | Comment marker for the GPU strided/uncoalesced vector-add kernel section; it documents the /performance lesson. |
| 0104 | GPU strided/uncoalesced vector-add kernel | `__global__ void vector_add_strided_kernel(const float* __restrict__ a,` | Declares a CUDA kernel for the GPU strided/uncoalesced vector-add kernel case; it runs on GPU. |
| 0105 | GPU strided/uncoalesced vector-add kernel | `                                          const float* __restrict__ b,` | Tells compiler pointers do not alias, which can help optimization. |
| 0106 | GPU strided/uncoalesced vector-add kernel | `                                          float* __restrict__ c,` | Tells compiler pointers do not alias, which can help optimization. |
| 0107 | GPU strided/uncoalesced vector-add kernel | `                                          size_t logical_n,` | C++/CUDA statement within the GPU strided/uncoalesced vector-add kernel section. |
| 0108 | GPU strided/uncoalesced vector-add kernel | `                                          int stride_elems) {` | C++/CUDA statement within the GPU strided/uncoalesced vector-add kernel section. |
| 0109 | GPU strided/uncoalesced vector-add kernel | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0110 | GPU strided/uncoalesced vector-add kernel | `    size_t grid_stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0111 | GPU strided/uncoalesced vector-add kernel | `    for (; i < logical_n; i += grid_stride) {` | Loop used by the GPU strided/uncoalesced vector-add kernel code to repeat work or traverse data. |
| 0112 | GPU strided/uncoalesced vector-add kernel | `        size_t j = i * static_cast<size_t>(stride_elems);` | Maps compact logical work to strided physical addresses, intentionally reducing coalescing. |
| 0113 | GPU strided/uncoalesced vector-add kernel | `        c[j] = a[j] + b[j];` | Performs same math as vector add but at strided memory locations. |
| 0114 | GPU strided/uncoalesced vector-add kernel | `    }` | Scope delimiter for the GPU strided/uncoalesced vector-add kernel code. |
| 0115 | GPU strided/uncoalesced vector-add kernel | `}` | Scope delimiter for the GPU strided/uncoalesced vector-add kernel code. |
| 0116 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0117 | host driver for vector/stride benchmark | `static Result run_gpu_vector_add(size_t n, int iters, bool strided) {` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0118 | host driver for vector/stride benchmark | `    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);` | Allocates host-side vector storage for the host driver for vector/stride benchmark benchmark. |
| 0119 | host driver for vector/stride benchmark | `    fill_vector(h_a); fill_vector(h_b);` | Initializes input data deterministically. |
| 0120 | host driver for vector/stride benchmark | `    for (size_t i = 0; i < n; ++i) h_ref[i] = h_a[i] + h_b[i];` | Builds CPU reference output used to validate GPU result. |
| 0121 | host driver for vector/stride benchmark | &nbsp; | Blank line separating the host driver for vector/stride benchmark code for readability. |
| 0122 | host driver for vector/stride benchmark | `    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);` | Allocates GPU memory through RAII wrapper. |
| 0123 | host driver for vector/stride benchmark | `    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0124 | host driver for vector/stride benchmark | `    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), d_b.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0125 | host driver for vector/stride benchmark | `    CUDA_CHECK(cudaMemset(d_c.get(), 0, d_c.bytes()));` | Initializes GPU memory, usually to avoid stale data or reset an output accumulator. |
| 0126 | host driver for vector/stride benchmark | &nbsp; | Blank line separating the host driver for vector/stride benchmark code for readability. |
| 0127 | host driver for vector/stride benchmark | `    int block = 256;` | Chooses 256 threads per block, a common starting point for CUDA kernels. |
| 0128 | host driver for vector/stride benchmark | `    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));` | Computes grid size needed to cover the problem. |
| 0129 | host driver for vector/stride benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0130 | host driver for vector/stride benchmark | &nbsp; | Blank line separating the host driver for vector/stride benchmark code for readability. |
| 0131 | host driver for vector/stride benchmark | `    if (!strided) {` | Conditional branch used by the host driver for vector/stride benchmark logic. |
| 0132 | host driver for vector/stride benchmark | `        vector_add_coalesced_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);` | Launches a CUDA kernel for the host driver for vector/stride benchmark case. |
| 0133 | host driver for vector/stride benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0134 | host driver for vector/stride benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0135 | host driver for vector/stride benchmark | `        for (int it = 0; it < iters; ++it) vector_add_coalesced_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);` | Launches a CUDA kernel for the host driver for vector/stride benchmark case. |
| 0136 | host driver for vector/stride benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0137 | host driver for vector/stride benchmark | `        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), d_c.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0138 | host driver for vector/stride benchmark | `        bool ok = approx_equal(h_c, h_ref);` | Stores validation status. |
| 0139 | host driver for vector/stride benchmark | `        double bytes = 3.0 * n * sizeof(float);` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0140 | host driver for vector/stride benchmark | `        return {"gpu_vector_add_coalesced", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0141 | host driver for vector/stride benchmark | `    } else {` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0142 | host driver for vector/stride benchmark | `        int stride = 8;` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0143 | host driver for vector/stride benchmark | `        size_t logical_n = n / stride;` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0144 | host driver for vector/stride benchmark | `        vector_add_strided_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), logical_n, stride);` | Launches a CUDA kernel for the host driver for vector/stride benchmark case. |
| 0145 | host driver for vector/stride benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0146 | host driver for vector/stride benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0147 | host driver for vector/stride benchmark | `        for (int it = 0; it < iters; ++it) vector_add_strided_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), logical_n, stride);` | Launches a CUDA kernel for the host driver for vector/stride benchmark case. |
| 0148 | host driver for vector/stride benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0149 | host driver for vector/stride benchmark | `        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), d_c.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0150 | host driver for vector/stride benchmark | `        bool ok = true;` | Stores validation status. |
| 0151 | host driver for vector/stride benchmark | `        for (size_t i = 0; i < logical_n; ++i) {` | Loop used by the host driver for vector/stride benchmark code to repeat work or traverse data. |
| 0152 | host driver for vector/stride benchmark | `            size_t j = i * stride;` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0153 | host driver for vector/stride benchmark | `            if (std::fabs(h_c[j] - h_ref[j]) > 1e-3f) { ok = false; break; }` | Computes absolute error for validation or relative-error checks. |
| 0154 | host driver for vector/stride benchmark | `        }` | Scope delimiter for the host driver for vector/stride benchmark code. |
| 0155 | host driver for vector/stride benchmark | `        // Requested useful bytes, not total address span. This exposes efficiency loss.` | Comment marker for the host driver for vector/stride benchmark section; it documents the /performance lesson. |
| 0156 | host driver for vector/stride benchmark | `        double bytes = 3.0 * logical_n * sizeof(float);` | C++/CUDA statement within the host driver for vector/stride benchmark section. |
| 0157 | host driver for vector/stride benchmark | `        return {"gpu_vector_add_strided", ms, bytes / (ms * 1.0e6), logical_n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0158 | host driver for vector/stride benchmark | `    }` | Scope delimiter for the host driver for vector/stride benchmark code. |
| 0159 | host driver for vector/stride benchmark | `}` | Scope delimiter for the host driver for vector/stride benchmark code. |
| 0160 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0161 | occupancy and register-pressure kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the occupancy and register-pressure kernels section; it documents the /performance lesson. |
| 0162 | occupancy and register-pressure kernels | `// 4. Occupancy/register pressure demo.  point: occupancy is not the final performance metric.` | Comment marker for the occupancy and register-pressure kernels section; it documents the /performance lesson. |
| 0163 | occupancy and register-pressure kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the occupancy and register-pressure kernels section; it documents the /performance lesson. |
| 0164 | occupancy and register-pressure kernels | `__global__ void low_register_kernel(const float* __restrict__ a, float* __restrict__ out, size_t n) {` | Declares a CUDA kernel for the occupancy and register-pressure kernels case; it runs on GPU. |
| 0165 | occupancy and register-pressure kernels | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0166 | occupancy and register-pressure kernels | `    size_t stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0167 | occupancy and register-pressure kernels | `    for (; i < n; i += stride) {` | Grid-stride loop lets each thread process multiple elements safely. |
| 0168 | occupancy and register-pressure kernels | `        float x = a[i];` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0169 | occupancy and register-pressure kernels | `        #pragma unroll 8` | Requests loop unrolling to change instruction scheduling/register behavior. |
| 0170 | occupancy and register-pressure kernels | `        for (int k = 0; k < 32; ++k) x = x * 1.000001f + 0.000001f;` | Loop used by the occupancy and register-pressure kernels code to repeat work or traverse data. |
| 0171 | occupancy and register-pressure kernels | `        out[i] = x;` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0172 | occupancy and register-pressure kernels | `    }` | Scope delimiter for the occupancy and register-pressure kernels code. |
| 0173 | occupancy and register-pressure kernels | `}` | Scope delimiter for the occupancy and register-pressure kernels code. |
| 0174 | occupancy and register-pressure kernels | &nbsp; | Blank line separating the occupancy and register-pressure kernels code for readability. |
| 0175 | occupancy and register-pressure kernels | `__global__ void high_register_kernel(const float* __restrict__ a, float* __restrict__ out, size_t n) {` | Declares a CUDA kernel for the occupancy and register-pressure kernels case; it runs on GPU. |
| 0176 | occupancy and register-pressure kernels | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0177 | occupancy and register-pressure kernels | `    size_t stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0178 | occupancy and register-pressure kernels | `    for (; i < n; i += stride) {` | Grid-stride loop lets each thread process multiple elements safely. |
| 0179 | occupancy and register-pressure kernels | `        float x = a[i];` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0180 | occupancy and register-pressure kernels | `        float r0=x, r1=x+1, r2=x+2, r3=x+3, r4=x+4, r5=x+5, r6=x+6, r7=x+7;` | Creates many live float variables to intentionally increase register pressure. |
| 0181 | occupancy and register-pressure kernels | `        float r8=x+8, r9=x+9, r10=x+10, r11=x+11, r12=x+12, r13=x+13, r14=x+14, r15=x+15;` | Creates many live float variables to intentionally increase register pressure. |
| 0182 | occupancy and register-pressure kernels | `        #pragma unroll 8` | Requests loop unrolling to change instruction scheduling/register behavior. |
| 0183 | occupancy and register-pressure kernels | `        for (int k = 0; k < 64; ++k) {` | Loop used by the occupancy and register-pressure kernels code to repeat work or traverse data. |
| 0184 | occupancy and register-pressure kernels | `            r0 = r0 * 1.000001f + r8;   r1 = r1 * 0.999999f + r9;` | Updates many live registers, modeling register-heavy/fused GPU code. |
| 0185 | occupancy and register-pressure kernels | `            r2 = r2 * 1.000002f + r10;  r3 = r3 * 0.999998f + r11;` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0186 | occupancy and register-pressure kernels | `            r4 = r4 * 1.000003f + r12;  r5 = r5 * 0.999997f + r13;` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0187 | occupancy and register-pressure kernels | `            r6 = r6 * 1.000004f + r14;  r7 = r7 * 0.999996f + r15;` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0188 | occupancy and register-pressure kernels | `            r8 += 0.000001f; r9 += 0.000002f; r10 += 0.000003f; r11 += 0.000004f;` | Updates many live registers, modeling register-heavy/fused GPU code. |
| 0189 | occupancy and register-pressure kernels | `            r12 += 0.000005f; r13 += 0.000006f; r14 += 0.000007f; r15 += 0.000008f;` | Updates many live registers, modeling register-heavy/fused GPU code. |
| 0190 | occupancy and register-pressure kernels | `        }` | Scope delimiter for the occupancy and register-pressure kernels code. |
| 0191 | occupancy and register-pressure kernels | `        out[i] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;` | C++/CUDA statement within the occupancy and register-pressure kernels section. |
| 0192 | occupancy and register-pressure kernels | `    }` | Scope delimiter for the occupancy and register-pressure kernels code. |
| 0193 | occupancy and register-pressure kernels | `}` | Scope delimiter for the occupancy and register-pressure kernels code. |
| 0194 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0195 | host driver for occupancy benchmark | `static Result run_occupancy(size_t n, int iters, bool high) {` | C++/CUDA statement within the host driver for occupancy benchmark section. |
| 0196 | host driver for occupancy benchmark | `    std::vector<float> h_a(n), h_out(n);` | Allocates host-side vector storage for the host driver for occupancy benchmark benchmark. |
| 0197 | host driver for occupancy benchmark | `    fill_vector(h_a);` | Initializes input data deterministically. |
| 0198 | host driver for occupancy benchmark | `    DeviceBuffer<float> d_a(n), d_out(n);` | Allocates GPU memory through RAII wrapper. |
| 0199 | host driver for occupancy benchmark | `    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0200 | host driver for occupancy benchmark | `    int block = 256;` | Chooses 256 threads per block, a common starting point for CUDA kernels. |
| 0201 | host driver for occupancy benchmark | `    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));` | Computes grid size needed to cover the problem. |
| 0202 | host driver for occupancy benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0203 | host driver for occupancy benchmark | `    if (high) high_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for occupancy benchmark case. |
| 0204 | host driver for occupancy benchmark | `    else low_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for occupancy benchmark case. |
| 0205 | host driver for occupancy benchmark | `    CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0206 | host driver for occupancy benchmark | `    timer.start();` | Starts GPU timing after warm-up. |
| 0207 | host driver for occupancy benchmark | `    for (int it = 0; it < iters; ++it) {` | Loop used by the host driver for occupancy benchmark code to repeat work or traverse data. |
| 0208 | host driver for occupancy benchmark | `        if (high) high_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for occupancy benchmark case. |
| 0209 | host driver for occupancy benchmark | `        else low_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for occupancy benchmark case. |
| 0210 | host driver for occupancy benchmark | `    }` | Scope delimiter for the host driver for occupancy benchmark code. |
| 0211 | host driver for occupancy benchmark | `    float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0212 | host driver for occupancy benchmark | `    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0213 | host driver for occupancy benchmark | `    bool ok = std::isfinite(h_out[n / 2]);` | Stores validation status. |
| 0214 | host driver for occupancy benchmark | `    return {high ? "high_register_occupancy_demo" : "low_register_occupancy_demo", ms, 0.0, n / (ms * 1.0e6), ok};` | Converts processed elements and milliseconds into GOP/s-style rate. |
| 0215 | host driver for occupancy benchmark | `}` | Scope delimiter for the host driver for occupancy benchmark code. |
| 0216 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0217 | reduction kernels: atomic vs warp/block staged | `// -------------------------------------------------------------------------------------------------` | Comment marker for the reduction kernels: atomic vs warp/block staged section; it documents the /performance lesson. |
| 0218 | reduction kernels: atomic vs warp/block staged | `// 5. Reduction: bad atomic vs block/warp reduction.  point: atomics/sync/staging matter.` | Comment marker for the reduction kernels: atomic vs warp/block staged section; it documents the /performance lesson. |
| 0219 | reduction kernels: atomic vs warp/block staged | `// -------------------------------------------------------------------------------------------------` | Comment marker for the reduction kernels: atomic vs warp/block staged section; it documents the /performance lesson. |
| 0220 | reduction kernels: atomic vs warp/block staged | `__global__ void reduce_atomic_bad_kernel(const float* __restrict__ in, float* out, size_t n) {` | Declares a CUDA kernel for the reduction kernels: atomic vs warp/block staged case; it runs on GPU. |
| 0221 | reduction kernels: atomic vs warp/block staged | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0222 | reduction kernels: atomic vs warp/block staged | `    size_t stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0223 | reduction kernels: atomic vs warp/block staged | `    float sum = 0.0f;` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0224 | reduction kernels: atomic vs warp/block staged | `    for (; i < n; i += stride) sum += in[i];` | Grid-stride loop lets each thread process multiple elements safely. |
| 0225 | reduction kernels: atomic vs warp/block staged | `    atomicAdd(out, sum);` | Uses global atomic addition; this can serialize under contention. |
| 0226 | reduction kernels: atomic vs warp/block staged | `}` | Scope delimiter for the reduction kernels: atomic vs warp/block staged code. |
| 0227 | reduction kernels: atomic vs warp/block staged | &nbsp; | Blank line separating the reduction kernels: atomic vs warp/block staged code for readability. |
| 0228 | reduction kernels: atomic vs warp/block staged | `__inline__ __device__ float warp_reduce_sum(float v) {` | Declares a GPU device helper function callable from kernels. |
| 0229 | reduction kernels: atomic vs warp/block staged | `    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);` | Uses warp shuffle to reduce values within a warp without shared memory. |
| 0230 | reduction kernels: atomic vs warp/block staged | `    return v;` | Returns from the current function in the reduction kernels: atomic vs warp/block staged section. |
| 0231 | reduction kernels: atomic vs warp/block staged | `}` | Scope delimiter for the reduction kernels: atomic vs warp/block staged code. |
| 0232 | reduction kernels: atomic vs warp/block staged | &nbsp; | Blank line separating the reduction kernels: atomic vs warp/block staged code for readability. |
| 0233 | reduction kernels: atomic vs warp/block staged | `__global__ void reduce_block_warp_kernel(const float* __restrict__ in, float* partial, size_t n) {` | Declares a CUDA kernel for the reduction kernels: atomic vs warp/block staged case; it runs on GPU. |
| 0234 | reduction kernels: atomic vs warp/block staged | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0235 | reduction kernels: atomic vs warp/block staged | `    size_t stride = blockDim.x * gridDim.x;` | Computes grid-stride step so all elements can be covered by limited grid size. |
| 0236 | reduction kernels: atomic vs warp/block staged | `    float sum = 0.0f;` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0237 | reduction kernels: atomic vs warp/block staged | `    for (; i < n; i += stride) sum += in[i];` | Grid-stride loop lets each thread process multiple elements safely. |
| 0238 | reduction kernels: atomic vs warp/block staged | &nbsp; | Blank line separating the reduction kernels: atomic vs warp/block staged code for readability. |
| 0239 | reduction kernels: atomic vs warp/block staged | `    sum = warp_reduce_sum(sum);` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0240 | reduction kernels: atomic vs warp/block staged | &nbsp; | Blank line separating the reduction kernels: atomic vs warp/block staged code for readability. |
| 0241 | reduction kernels: atomic vs warp/block staged | `    __shared__ float warp_sums[32];` | Allocates shared memory within one thread block. |
| 0242 | reduction kernels: atomic vs warp/block staged | `    int lane = threadIdx.x & 31;` | Computes lane id inside a 32-thread warp. |
| 0243 | reduction kernels: atomic vs warp/block staged | `    int warp_id = threadIdx.x >> 5;` | Computes warp id inside the block. |
| 0244 | reduction kernels: atomic vs warp/block staged | `    if (lane == 0) warp_sums[warp_id] = sum;` | Conditional branch used by the reduction kernels: atomic vs warp/block staged logic. |
| 0245 | reduction kernels: atomic vs warp/block staged | `    __syncthreads();` | Synchronizes all threads in the block before reading shared data. |
| 0246 | reduction kernels: atomic vs warp/block staged | &nbsp; | Blank line separating the reduction kernels: atomic vs warp/block staged code for readability. |
| 0247 | reduction kernels: atomic vs warp/block staged | `    float block_sum = 0.0f;` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0248 | reduction kernels: atomic vs warp/block staged | `    if (warp_id == 0) {` | Conditional branch used by the reduction kernels: atomic vs warp/block staged logic. |
| 0249 | reduction kernels: atomic vs warp/block staged | `        int num_warps = (blockDim.x + 31) >> 5;` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0250 | reduction kernels: atomic vs warp/block staged | `        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0251 | reduction kernels: atomic vs warp/block staged | `        block_sum = warp_reduce_sum(block_sum);` | C++/CUDA statement within the reduction kernels: atomic vs warp/block staged section. |
| 0252 | reduction kernels: atomic vs warp/block staged | `        if (lane == 0) partial[blockIdx.x] = block_sum;` | Writes one partial reduction result per block. |
| 0253 | reduction kernels: atomic vs warp/block staged | `    }` | Scope delimiter for the reduction kernels: atomic vs warp/block staged code. |
| 0254 | reduction kernels: atomic vs warp/block staged | `}` | Scope delimiter for the reduction kernels: atomic vs warp/block staged code. |
| 0255 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0256 | host driver for reduction benchmark | `static Result run_reduction(size_t n, int iters, bool atomic_bad) {` | C++/CUDA statement within the host driver for reduction benchmark section. |
| 0257 | host driver for reduction benchmark | `    std::vector<float> h_in(n);` | Allocates host-side vector storage for the host driver for reduction benchmark benchmark. |
| 0258 | host driver for reduction benchmark | `    fill_vector(h_in);` | Initializes input data deterministically. |
| 0259 | host driver for reduction benchmark | `    double ref = std::accumulate(h_in.begin(), h_in.end(), 0.0);` | Computes CPU-side reference or final accumulation for validation. |
| 0260 | host driver for reduction benchmark | `    DeviceBuffer<float> d_in(n);` | Allocates GPU memory through RAII wrapper. |
| 0261 | host driver for reduction benchmark | `    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), d_in.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0262 | host driver for reduction benchmark | `    int block = 256;` | Chooses 256 threads per block, a common starting point for CUDA kernels. |
| 0263 | host driver for reduction benchmark | `    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));` | Computes grid size needed to cover the problem. |
| 0264 | host driver for reduction benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0265 | host driver for reduction benchmark | `    float sum = 0.0f;` | C++/CUDA statement within the host driver for reduction benchmark section. |
| 0266 | host driver for reduction benchmark | &nbsp; | Blank line separating the host driver for reduction benchmark code for readability. |
| 0267 | host driver for reduction benchmark | `    if (atomic_bad) {` | Conditional branch used by the host driver for reduction benchmark logic. |
| 0268 | host driver for reduction benchmark | `        DeviceBuffer<float> d_out(1);` | Allocates GPU memory through RAII wrapper. |
| 0269 | host driver for reduction benchmark | `        CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(float)));` | Initializes GPU memory, usually to avoid stale data or reset an output accumulator. |
| 0270 | host driver for reduction benchmark | `        reduce_atomic_bad_kernel<<<grid, block>>>(d_in.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for reduction benchmark case. |
| 0271 | host driver for reduction benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0272 | host driver for reduction benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0273 | host driver for reduction benchmark | `        for (int it = 0; it < iters; ++it) {` | Loop used by the host driver for reduction benchmark code to repeat work or traverse data. |
| 0274 | host driver for reduction benchmark | `            CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(float)));` | Initializes GPU memory, usually to avoid stale data or reset an output accumulator. |
| 0275 | host driver for reduction benchmark | `            reduce_atomic_bad_kernel<<<grid, block>>>(d_in.get(), d_out.get(), n);` | Launches a CUDA kernel for the host driver for reduction benchmark case. |
| 0276 | host driver for reduction benchmark | `        }` | Scope delimiter for the host driver for reduction benchmark code. |
| 0277 | host driver for reduction benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0278 | host driver for reduction benchmark | `        CUDA_CHECK(cudaMemcpy(&sum, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0279 | host driver for reduction benchmark | `        bool ok = std::fabs(sum - static_cast<float>(ref)) / std::max(1.0f, std::fabs(static_cast<float>(ref))) < 1e-2f;` | Stores validation status. |
| 0280 | host driver for reduction benchmark | `        double bytes = n * sizeof(float);` | C++/CUDA statement within the host driver for reduction benchmark section. |
| 0281 | host driver for reduction benchmark | `        return {"reduce_atomic_bad", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0282 | host driver for reduction benchmark | `    } else {` | C++/CUDA statement within the host driver for reduction benchmark section. |
| 0283 | host driver for reduction benchmark | `        DeviceBuffer<float> d_partial(grid);` | Allocates GPU memory through RAII wrapper. |
| 0284 | host driver for reduction benchmark | `        std::vector<float> h_partial(grid);` | Allocates host-side vector storage for the host driver for reduction benchmark benchmark. |
| 0285 | host driver for reduction benchmark | `        reduce_block_warp_kernel<<<grid, block>>>(d_in.get(), d_partial.get(), n);` | Launches a CUDA kernel for the host driver for reduction benchmark case. |
| 0286 | host driver for reduction benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0287 | host driver for reduction benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0288 | host driver for reduction benchmark | `        for (int it = 0; it < iters; ++it) reduce_block_warp_kernel<<<grid, block>>>(d_in.get(), d_partial.get(), n);` | Launches a CUDA kernel for the host driver for reduction benchmark case. |
| 0289 | host driver for reduction benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0290 | host driver for reduction benchmark | `        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial.get(), grid * sizeof(float), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0291 | host driver for reduction benchmark | `        double got = std::accumulate(h_partial.begin(), h_partial.end(), 0.0);` | Computes CPU-side reference or final accumulation for validation. |
| 0292 | host driver for reduction benchmark | `        bool ok = std::fabs(got - ref) / std::max(1.0, std::fabs(ref)) < 1e-2;` | Stores validation status. |
| 0293 | host driver for reduction benchmark | `        double bytes = n * sizeof(float) + grid * sizeof(float);` | C++/CUDA statement within the host driver for reduction benchmark section. |
| 0294 | host driver for reduction benchmark | `        return {"reduce_block_warp", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0295 | host driver for reduction benchmark | `    }` | Scope delimiter for the host driver for reduction benchmark code. |
| 0296 | host driver for reduction benchmark | `}` | Scope delimiter for the host driver for reduction benchmark code. |
| 0297 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0298 | matrix transpose kernels: naive vs tiled/shared-memory | `// -------------------------------------------------------------------------------------------------` | Comment marker for the matrix transpose kernels: naive vs tiled/shared-memory section; it documents the /performance lesson. |
| 0299 | matrix transpose kernels: naive vs tiled/shared-memory | `// 6. Matrix transpose: naive vs shared-memory tiled+padded.  point: coalescing and bank conflicts.` | Comment marker for the matrix transpose kernels: naive vs tiled/shared-memory section; it documents the /performance lesson. |
| 0300 | matrix transpose kernels: naive vs tiled/shared-memory | `// -------------------------------------------------------------------------------------------------` | Comment marker for the matrix transpose kernels: naive vs tiled/shared-memory section; it documents the /performance lesson. |
| 0301 | matrix transpose kernels: naive vs tiled/shared-memory | `template <int TILE_DIM, int BLOCK_ROWS>` | Makes transpose tile size a compile-time parameter for optimization. |
| 0302 | matrix transpose kernels: naive vs tiled/shared-memory | `__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {` | Declares a CUDA kernel for the matrix transpose kernels: naive vs tiled/shared-memory case; it runs on GPU. |
| 0303 | matrix transpose kernels: naive vs tiled/shared-memory | `    int x = blockIdx.x * TILE_DIM + threadIdx.x;` | C++/CUDA statement within the matrix transpose kernels: naive vs tiled/shared-memory section. |
| 0304 | matrix transpose kernels: naive vs tiled/shared-memory | `    int y = blockIdx.y * TILE_DIM + threadIdx.y;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0305 | matrix transpose kernels: naive vs tiled/shared-memory | `    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {` | Loop used by the matrix transpose kernels: naive vs tiled/shared-memory code to repeat work or traverse data. |
| 0306 | matrix transpose kernels: naive vs tiled/shared-memory | `        if (x < width && (y + j) < height) out[x * height + (y + j)] = in[(y + j) * width + x];` | Naive transpose store; likely uncoalesced for one side of the transpose. |
| 0307 | matrix transpose kernels: naive vs tiled/shared-memory | `    }` | Scope delimiter for the matrix transpose kernels: naive vs tiled/shared-memory code. |
| 0308 | matrix transpose kernels: naive vs tiled/shared-memory | `}` | Scope delimiter for the matrix transpose kernels: naive vs tiled/shared-memory code. |
| 0309 | matrix transpose kernels: naive vs tiled/shared-memory | &nbsp; | Blank line separating the matrix transpose kernels: naive vs tiled/shared-memory code for readability. |
| 0310 | matrix transpose kernels: naive vs tiled/shared-memory | `template <int TILE_DIM, int BLOCK_ROWS>` | Makes transpose tile size a compile-time parameter for optimization. |
| 0311 | matrix transpose kernels: naive vs tiled/shared-memory | `__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {` | Declares a CUDA kernel for the matrix transpose kernels: naive vs tiled/shared-memory case; it runs on GPU. |
| 0312 | matrix transpose kernels: naive vs tiled/shared-memory | `    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids shared-memory bank conflicts on column reads` | Allocates shared memory within one thread block. |
| 0313 | matrix transpose kernels: naive vs tiled/shared-memory | &nbsp; | Blank line separating the matrix transpose kernels: naive vs tiled/shared-memory code for readability. |
| 0314 | matrix transpose kernels: naive vs tiled/shared-memory | `    int x = blockIdx.x * TILE_DIM + threadIdx.x;` | C++/CUDA statement within the matrix transpose kernels: naive vs tiled/shared-memory section. |
| 0315 | matrix transpose kernels: naive vs tiled/shared-memory | `    int y = blockIdx.y * TILE_DIM + threadIdx.y;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0316 | matrix transpose kernels: naive vs tiled/shared-memory | &nbsp; | Blank line separating the matrix transpose kernels: naive vs tiled/shared-memory code for readability. |
| 0317 | matrix transpose kernels: naive vs tiled/shared-memory | `    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {` | Loop used by the matrix transpose kernels: naive vs tiled/shared-memory code to repeat work or traverse data. |
| 0318 | matrix transpose kernels: naive vs tiled/shared-memory | `        if (x < width && (y + j) < height) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0319 | matrix transpose kernels: naive vs tiled/shared-memory | `    }` | Scope delimiter for the matrix transpose kernels: naive vs tiled/shared-memory code. |
| 0320 | matrix transpose kernels: naive vs tiled/shared-memory | `    __syncthreads();` | Synchronizes all threads in the block before reading shared data. |
| 0321 | matrix transpose kernels: naive vs tiled/shared-memory | &nbsp; | Blank line separating the matrix transpose kernels: naive vs tiled/shared-memory code for readability. |
| 0322 | matrix transpose kernels: naive vs tiled/shared-memory | `    x = blockIdx.y * TILE_DIM + threadIdx.x;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0323 | matrix transpose kernels: naive vs tiled/shared-memory | `    y = blockIdx.x * TILE_DIM + threadIdx.y;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0324 | matrix transpose kernels: naive vs tiled/shared-memory | &nbsp; | Blank line separating the matrix transpose kernels: naive vs tiled/shared-memory code for readability. |
| 0325 | matrix transpose kernels: naive vs tiled/shared-memory | `    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {` | Loop used by the matrix transpose kernels: naive vs tiled/shared-memory code to repeat work or traverse data. |
| 0326 | matrix transpose kernels: naive vs tiled/shared-memory | `        if (x < height && (y + j) < width) out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0327 | matrix transpose kernels: naive vs tiled/shared-memory | `    }` | Scope delimiter for the matrix transpose kernels: naive vs tiled/shared-memory code. |
| 0328 | matrix transpose kernels: naive vs tiled/shared-memory | `}` | Scope delimiter for the matrix transpose kernels: naive vs tiled/shared-memory code. |
| 0329 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0330 | host driver for transpose benchmark | `static Result run_transpose(int width, int height, int iters, bool tiled) {` | C++/CUDA statement within the host driver for transpose benchmark section. |
| 0331 | host driver for transpose benchmark | `    size_t n = static_cast<size_t>(width) * height;` | C++/CUDA statement within the host driver for transpose benchmark section. |
| 0332 | host driver for transpose benchmark | `    std::vector<float> h_in(n), h_out(n), h_ref(n);` | Allocates host-side vector storage for the host driver for transpose benchmark benchmark. |
| 0333 | host driver for transpose benchmark | `    fill_vector(h_in);` | Initializes input data deterministically. |
| 0334 | host driver for transpose benchmark | `    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) h_ref[x * height + y] = h_in[y * width + x];` | Builds CPU reference output used to validate GPU result. |
| 0335 | host driver for transpose benchmark | `    DeviceBuffer<float> d_in(n), d_out(n);` | Allocates GPU memory through RAII wrapper. |
| 0336 | host driver for transpose benchmark | `    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), d_in.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0337 | host driver for transpose benchmark | &nbsp; | Blank line separating the host driver for transpose benchmark code for readability. |
| 0338 | host driver for transpose benchmark | `    constexpr int TILE = 32;` | C++/CUDA statement within the host driver for transpose benchmark section. |
| 0339 | host driver for transpose benchmark | `    constexpr int ROWS = 8;` | C++/CUDA statement within the host driver for transpose benchmark section. |
| 0340 | host driver for transpose benchmark | `    dim3 block(TILE, ROWS);` | Defines 2D CUDA block shape. |
| 0341 | host driver for transpose benchmark | `    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);` | Computes grid size needed to cover the problem. |
| 0342 | host driver for transpose benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0343 | host driver for transpose benchmark | `    if (tiled) transpose_tiled_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);` | Launches a CUDA kernel for the host driver for transpose benchmark case. |
| 0344 | host driver for transpose benchmark | `    else transpose_naive_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);` | Launches a CUDA kernel for the host driver for transpose benchmark case. |
| 0345 | host driver for transpose benchmark | `    CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0346 | host driver for transpose benchmark | `    timer.start();` | Starts GPU timing after warm-up. |
| 0347 | host driver for transpose benchmark | `    for (int it = 0; it < iters; ++it) {` | Loop used by the host driver for transpose benchmark code to repeat work or traverse data. |
| 0348 | host driver for transpose benchmark | `        if (tiled) transpose_tiled_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);` | Launches a CUDA kernel for the host driver for transpose benchmark case. |
| 0349 | host driver for transpose benchmark | `        else transpose_naive_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);` | Launches a CUDA kernel for the host driver for transpose benchmark case. |
| 0350 | host driver for transpose benchmark | `    }` | Scope delimiter for the host driver for transpose benchmark code. |
| 0351 | host driver for transpose benchmark | `    float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0352 | host driver for transpose benchmark | `    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0353 | host driver for transpose benchmark | `    bool ok = approx_equal(h_out, h_ref);` | Stores validation status. |
| 0354 | host driver for transpose benchmark | `    double bytes = 2.0 * n * sizeof(float);` | C++/CUDA statement within the host driver for transpose benchmark section. |
| 0355 | host driver for transpose benchmark | `    return {tiled ? "transpose_tiled_padded" : "transpose_naive", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0356 | host driver for transpose benchmark | `}` | Scope delimiter for the host driver for transpose benchmark code. |
| 0357 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0358 | shared-memory bank-conflict kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the shared-memory bank-conflict kernels section; it documents the /performance lesson. |
| 0359 | shared-memory bank-conflict kernels | `// 7. Shared memory bank-conflict micro demo.  point: padding changes bank mapping.` | Comment marker for the shared-memory bank-conflict kernels section; it documents the /performance lesson. |
| 0360 | shared-memory bank-conflict kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the shared-memory bank-conflict kernels section; it documents the /performance lesson. |
| 0361 | shared-memory bank-conflict kernels | `__global__ void shared_bank_conflict_kernel(float* out, int reps) {` | Declares a CUDA kernel for the shared-memory bank-conflict kernels case; it runs on GPU. |
| 0362 | shared-memory bank-conflict kernels | `    __shared__ float tile[32][32];` | Allocates shared memory within one thread block. |
| 0363 | shared-memory bank-conflict kernels | `    int tx = threadIdx.x;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0364 | shared-memory bank-conflict kernels | `    int ty = threadIdx.y;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0365 | shared-memory bank-conflict kernels | `    float v = static_cast<float>(tx + ty);` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0366 | shared-memory bank-conflict kernels | `    tile[ty][tx] = v;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0367 | shared-memory bank-conflict kernels | `    __syncthreads();` | Synchronizes all threads in the block before reading shared data. |
| 0368 | shared-memory bank-conflict kernels | `    float sum = 0.0f;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0369 | shared-memory bank-conflict kernels | `    for (int r = 0; r < reps; ++r) sum += tile[tx][ty]; // column-style read can conflict` | Column-style shared-memory read used to demonstrate bank conflicts. |
| 0370 | shared-memory bank-conflict kernels | `    out[ty * 32 + tx] = sum;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0371 | shared-memory bank-conflict kernels | `}` | Scope delimiter for the shared-memory bank-conflict kernels code. |
| 0372 | shared-memory bank-conflict kernels | &nbsp; | Blank line separating the shared-memory bank-conflict kernels code for readability. |
| 0373 | shared-memory bank-conflict kernels | `__global__ void shared_bank_padded_kernel(float* out, int reps) {` | Declares a CUDA kernel for the shared-memory bank-conflict kernels case; it runs on GPU. |
| 0374 | shared-memory bank-conflict kernels | `    __shared__ float tile[32][33];` | Allocates shared memory within one thread block. |
| 0375 | shared-memory bank-conflict kernels | `    int tx = threadIdx.x;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0376 | shared-memory bank-conflict kernels | `    int ty = threadIdx.y;` | Computes 2D block/thread coordinate for matrix/tile access. |
| 0377 | shared-memory bank-conflict kernels | `    float v = static_cast<float>(tx + ty);` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0378 | shared-memory bank-conflict kernels | `    tile[ty][tx] = v;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0379 | shared-memory bank-conflict kernels | `    __syncthreads();` | Synchronizes all threads in the block before reading shared data. |
| 0380 | shared-memory bank-conflict kernels | `    float sum = 0.0f;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0381 | shared-memory bank-conflict kernels | `    for (int r = 0; r < reps; ++r) sum += tile[tx][ty];` | Column-style shared-memory read used to demonstrate bank conflicts. |
| 0382 | shared-memory bank-conflict kernels | `    out[ty * 32 + tx] = sum;` | C++/CUDA statement within the shared-memory bank-conflict kernels section. |
| 0383 | shared-memory bank-conflict kernels | `}` | Scope delimiter for the shared-memory bank-conflict kernels code. |
| 0384 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0385 | host driver for bank-conflict benchmark | `static Result run_bank_conflict(int iters, bool padded) {` | C++/CUDA statement within the host driver for bank-conflict benchmark section. |
| 0386 | host driver for bank-conflict benchmark | `    DeviceBuffer<float> d_out(32 * 32);` | Allocates GPU memory through RAII wrapper. |
| 0387 | host driver for bank-conflict benchmark | `    dim3 block(32, 32);` | Defines 2D CUDA block shape. |
| 0388 | host driver for bank-conflict benchmark | `    int reps = 512;` | C++/CUDA statement within the host driver for bank-conflict benchmark section. |
| 0389 | host driver for bank-conflict benchmark | `    if (padded) shared_bank_padded_kernel<<<1, block>>>(d_out.get(), reps);` | Launches a CUDA kernel for the host driver for bank-conflict benchmark case. |
| 0390 | host driver for bank-conflict benchmark | `    else shared_bank_conflict_kernel<<<1, block>>>(d_out.get(), reps);` | Launches a CUDA kernel for the host driver for bank-conflict benchmark case. |
| 0391 | host driver for bank-conflict benchmark | `    CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0392 | host driver for bank-conflict benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0393 | host driver for bank-conflict benchmark | `    timer.start();` | Starts GPU timing after warm-up. |
| 0394 | host driver for bank-conflict benchmark | `    for (int it = 0; it < iters; ++it) {` | Loop used by the host driver for bank-conflict benchmark code to repeat work or traverse data. |
| 0395 | host driver for bank-conflict benchmark | `        if (padded) shared_bank_padded_kernel<<<1, block>>>(d_out.get(), reps);` | Launches a CUDA kernel for the host driver for bank-conflict benchmark case. |
| 0396 | host driver for bank-conflict benchmark | `        else shared_bank_conflict_kernel<<<1, block>>>(d_out.get(), reps);` | Launches a CUDA kernel for the host driver for bank-conflict benchmark case. |
| 0397 | host driver for bank-conflict benchmark | `    }` | Scope delimiter for the host driver for bank-conflict benchmark code. |
| 0398 | host driver for bank-conflict benchmark | `    float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0399 | host driver for bank-conflict benchmark | `    std::vector<float> h(32 * 32);` | Allocates host-side vector storage for the host driver for bank-conflict benchmark benchmark. |
| 0400 | host driver for bank-conflict benchmark | `    CUDA_CHECK(cudaMemcpy(h.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0401 | host driver for bank-conflict benchmark | `    bool ok = std::isfinite(h[17]);` | Stores validation status. |
| 0402 | host driver for bank-conflict benchmark | `    return {padded ? "shared_bank_padded" : "shared_bank_conflict", ms, 0.0, 0.0, ok};` | Returns from the current function in the host driver for bank-conflict benchmark section. |
| 0403 | host driver for bank-conflict benchmark | `}` | Scope delimiter for the host driver for bank-conflict benchmark code. |
| 0404 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0405 | AoS vs SoA data-layout kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the AoS vs SoA data-layout kernels section; it documents the /performance lesson. |
| 0406 | AoS vs SoA data-layout kernels | `// 8. AoS vs SoA data layout.  point: memory layout can dominate simple arithmetic.` | Comment marker for the AoS vs SoA data-layout kernels section; it documents the /performance lesson. |
| 0407 | AoS vs SoA data-layout kernels | `// -------------------------------------------------------------------------------------------------` | Comment marker for the AoS vs SoA data-layout kernels section; it documents the /performance lesson. |
| 0408 | AoS vs SoA data-layout kernels | `struct ParticleAoS { float x, y, z, vx, vy, vz; };` | Defines Array-of-Structs layout: all fields for one particle are adjacent. |
| 0409 | AoS vs SoA data-layout kernels | &nbsp; | Blank line separating the AoS vs SoA data-layout kernels code for readability. |
| 0410 | AoS vs SoA data-layout kernels | `__global__ void update_aos_kernel(ParticleAoS* p, float dt, size_t n) {` | Declares a CUDA kernel for the AoS vs SoA data-layout kernels case; it runs on GPU. |
| 0411 | AoS vs SoA data-layout kernels | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0412 | AoS vs SoA data-layout kernels | `    if (i < n) p[i].x += p[i].vx * dt;` | Updates x field in AoS layout; adjacent threads access x fields separated by struct size. |
| 0413 | AoS vs SoA data-layout kernels | `}` | Scope delimiter for the AoS vs SoA data-layout kernels code. |
| 0414 | AoS vs SoA data-layout kernels | &nbsp; | Blank line separating the AoS vs SoA data-layout kernels code for readability. |
| 0415 | AoS vs SoA data-layout kernels | `__global__ void update_soa_kernel(float* x, const float* vx, float dt, size_t n) {` | Declares a CUDA kernel for the AoS vs SoA data-layout kernels case; it runs on GPU. |
| 0416 | AoS vs SoA data-layout kernels | `    size_t i = blockIdx.x * blockDim.x + threadIdx.x;` | Computes unique global thread index for 1D CUDA work distribution. |
| 0417 | AoS vs SoA data-layout kernels | `    if (i < n) x[i] += vx[i] * dt;` | Updates x in SoA layout; adjacent threads access contiguous x and vx arrays. |
| 0418 | AoS vs SoA data-layout kernels | `}` | Scope delimiter for the AoS vs SoA data-layout kernels code. |
| 0419 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0420 | host driver for AoS/SoA benchmark | `static Result run_aos_soa(size_t n, int iters, bool soa) {` | C++/CUDA statement within the host driver for AoS/SoA benchmark section. |
| 0421 | host driver for AoS/SoA benchmark | `    int block = 256;` | Chooses 256 threads per block, a common starting point for CUDA kernels. |
| 0422 | host driver for AoS/SoA benchmark | `    int grid = static_cast<int>((n + block - 1) / block);` | Computes grid size needed to cover the problem. |
| 0423 | host driver for AoS/SoA benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0424 | host driver for AoS/SoA benchmark | `    bool ok = true;` | Stores validation status. |
| 0425 | host driver for AoS/SoA benchmark | `    if (!soa) {` | Conditional branch used by the host driver for AoS/SoA benchmark logic. |
| 0426 | host driver for AoS/SoA benchmark | `        std::vector<ParticleAoS> h(n);` | Allocates host-side vector storage for the host driver for AoS/SoA benchmark benchmark. |
| 0427 | host driver for AoS/SoA benchmark | `        for (size_t i = 0; i < n; ++i) h[i] = {1,2,3,0.5f,0.6f,0.7f};` | Loop used by the host driver for AoS/SoA benchmark code to repeat work or traverse data. |
| 0428 | host driver for AoS/SoA benchmark | `        DeviceBuffer<ParticleAoS> d(n);` | Allocates GPU memory through RAII wrapper. |
| 0429 | host driver for AoS/SoA benchmark | `        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0430 | host driver for AoS/SoA benchmark | `        update_aos_kernel<<<grid, block>>>(d.get(), 0.1f, n);` | Launches a CUDA kernel for the host driver for AoS/SoA benchmark case. |
| 0431 | host driver for AoS/SoA benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0432 | host driver for AoS/SoA benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0433 | host driver for AoS/SoA benchmark | `        for (int it = 0; it < iters; ++it) update_aos_kernel<<<grid, block>>>(d.get(), 0.1f, n);` | Launches a CUDA kernel for the host driver for AoS/SoA benchmark case. |
| 0434 | host driver for AoS/SoA benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0435 | host driver for AoS/SoA benchmark | `        CUDA_CHECK(cudaMemcpy(h.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0436 | host driver for AoS/SoA benchmark | `        ok = std::isfinite(h[n/2].x);` | Sanity-checks that output is a valid finite number. |
| 0437 | host driver for AoS/SoA benchmark | `        double useful_bytes = 2.0 * n * sizeof(float) + n * sizeof(float);` | C++/CUDA statement within the host driver for AoS/SoA benchmark section. |
| 0438 | host driver for AoS/SoA benchmark | `        return {"aos_update_x", ms, useful_bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0439 | host driver for AoS/SoA benchmark | `    } else {` | C++/CUDA statement within the host driver for AoS/SoA benchmark section. |
| 0440 | host driver for AoS/SoA benchmark | `        std::vector<float> h_x(n, 1.0f), h_vx(n, 0.5f);` | Allocates host-side vector storage for the host driver for AoS/SoA benchmark benchmark. |
| 0441 | host driver for AoS/SoA benchmark | `        DeviceBuffer<float> d_x(n), d_vx(n);` | Allocates GPU memory through RAII wrapper. |
| 0442 | host driver for AoS/SoA benchmark | `        CUDA_CHECK(cudaMemcpy(d_x.get(), h_x.data(), d_x.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0443 | host driver for AoS/SoA benchmark | `        CUDA_CHECK(cudaMemcpy(d_vx.get(), h_vx.data(), d_vx.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0444 | host driver for AoS/SoA benchmark | `        update_soa_kernel<<<grid, block>>>(d_x.get(), d_vx.get(), 0.1f, n);` | Launches a CUDA kernel for the host driver for AoS/SoA benchmark case. |
| 0445 | host driver for AoS/SoA benchmark | `        CUDA_KERNEL_CHECK();` | Checks kernel launch and execution errors after warm-up. |
| 0446 | host driver for AoS/SoA benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0447 | host driver for AoS/SoA benchmark | `        for (int it = 0; it < iters; ++it) update_soa_kernel<<<grid, block>>>(d_x.get(), d_vx.get(), 0.1f, n);` | Launches a CUDA kernel for the host driver for AoS/SoA benchmark case. |
| 0448 | host driver for AoS/SoA benchmark | `        float ms = timer.stop_ms() / iters;` | Stores average runtime in milliseconds. |
| 0449 | host driver for AoS/SoA benchmark | `        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.get(), d_x.bytes(), cudaMemcpyDeviceToHost));` | Transfers data between host and device for setup or validation. |
| 0450 | host driver for AoS/SoA benchmark | `        ok = std::isfinite(h_x[n/2]);` | Sanity-checks that output is a valid finite number. |
| 0451 | host driver for AoS/SoA benchmark | `        double bytes = 3.0 * n * sizeof(float);` | C++/CUDA statement within the host driver for AoS/SoA benchmark section. |
| 0452 | host driver for AoS/SoA benchmark | `        return {"soa_update_x", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};` | Converts useful bytes and milliseconds into GB/s. |
| 0453 | host driver for AoS/SoA benchmark | `    }` | Scope delimiter for the host driver for AoS/SoA benchmark code. |
| 0454 | host driver for AoS/SoA benchmark | `}` | Scope delimiter for the host driver for AoS/SoA benchmark code. |
| 0455 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0456 | pageable vs pinned host/device copy benchmark | `// -------------------------------------------------------------------------------------------------` | Comment marker for the pageable vs pinned host/device copy benchmark section; it documents the /performance lesson. |
| 0457 | pageable vs pinned host/device copy benchmark | `// 9. Host/device copy: pageable vs pinned.  point: end-to-end pipeline may bottleneck outside kernel.` | Comment marker for the pageable vs pinned host/device copy benchmark section; it documents the /performance lesson. |
| 0458 | pageable vs pinned host/device copy benchmark | `// -------------------------------------------------------------------------------------------------` | Comment marker for the pageable vs pinned host/device copy benchmark section; it documents the /performance lesson. |
| 0459 | pageable vs pinned host/device copy benchmark | `static Result run_copy(size_t n, int iters, bool pinned) {` | C++/CUDA statement within the pageable vs pinned host/device copy benchmark section. |
| 0460 | pageable vs pinned host/device copy benchmark | `    DeviceBuffer<float> d(n);` | Allocates GPU memory through RAII wrapper. |
| 0461 | pageable vs pinned host/device copy benchmark | `    GpuTimer timer;` | Creates CUDA event timer for GPU timing. |
| 0462 | pageable vs pinned host/device copy benchmark | `    float ms = 0.0f;` | Stores average runtime in milliseconds. |
| 0463 | pageable vs pinned host/device copy benchmark | `    if (!pinned) {` | Conditional branch used by the pageable vs pinned host/device copy benchmark logic. |
| 0464 | pageable vs pinned host/device copy benchmark | `        std::vector<float> h(n, 1.0f);` | Allocates host-side vector storage for the pageable vs pinned host/device copy benchmark benchmark. |
| 0465 | pageable vs pinned host/device copy benchmark | `        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0466 | pageable vs pinned host/device copy benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0467 | pageable vs pinned host/device copy benchmark | `        for (int it = 0; it < iters; ++it) CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0468 | pageable vs pinned host/device copy benchmark | `        ms = timer.stop_ms() / iters;` | Stops GPU timer and computes average time. |
| 0469 | pageable vs pinned host/device copy benchmark | `    } else {` | C++/CUDA statement within the pageable vs pinned host/device copy benchmark section. |
| 0470 | pageable vs pinned host/device copy benchmark | `        PinnedHostBuffer<float> h(n);` | Allocates pinned host memory for copy-bandwidth comparison. |
| 0471 | pageable vs pinned host/device copy benchmark | `        for (size_t i = 0; i < n; ++i) h[i] = 1.0f;` | Loop used by the pageable vs pinned host/device copy benchmark code to repeat work or traverse data. |
| 0472 | pageable vs pinned host/device copy benchmark | `        CUDA_CHECK(cudaMemcpy(d.get(), h.get(), d.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0473 | pageable vs pinned host/device copy benchmark | `        timer.start();` | Starts GPU timing after warm-up. |
| 0474 | pageable vs pinned host/device copy benchmark | `        for (int it = 0; it < iters; ++it) CUDA_CHECK(cudaMemcpy(d.get(), h.get(), d.bytes(), cudaMemcpyHostToDevice));` | Transfers data between host and device for setup or validation. |
| 0475 | pageable vs pinned host/device copy benchmark | `        ms = timer.stop_ms() / iters;` | Stops GPU timer and computes average time. |
| 0476 | pageable vs pinned host/device copy benchmark | `    }` | Scope delimiter for the pageable vs pinned host/device copy benchmark code. |
| 0477 | pageable vs pinned host/device copy benchmark | `    double bytes = n * sizeof(float);` | C++/CUDA statement within the pageable vs pinned host/device copy benchmark section. |
| 0478 | pageable vs pinned host/device copy benchmark | `    return {pinned ? "h2d_copy_pinned" : "h2d_copy_pageable", ms, bytes / (ms * 1.0e6), 0.0, true};` | Converts useful bytes and milliseconds into GB/s. |
| 0479 | pageable vs pinned host/device copy benchmark | `}` | Scope delimiter for the pageable vs pinned host/device copy benchmark code. |
| 0480 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0481 | GPU device information printing | `static void print_device_info() {` | C++/CUDA statement within the GPU device information printing section. |
| 0482 | GPU device information printing | `    int dev = 0;` | C++/CUDA statement within the GPU device information printing section. |
| 0483 | GPU device information printing | `    CUDA_CHECK(cudaGetDevice(&dev));` | Gets the active CUDA device id. |
| 0484 | GPU device information printing | `    cudaDeviceProp prop{};` | C++/CUDA statement within the GPU device information printing section. |
| 0485 | GPU device information printing | `    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));` | Queries GPU hardware properties such as name, SM count, memory, and warp size. |
| 0486 | GPU device information printing | `    std::cout << "GPU: " << prop.name << "\n";` | C++/CUDA statement within the GPU device information printing section. |
| 0487 | GPU device information printing | `    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";` | C++/CUDA statement within the GPU device information printing section. |
| 0488 | GPU device information printing | `    std::cout << "SM count: " << prop.multiProcessorCount << "\n";` | C++/CUDA statement within the GPU device information printing section. |
| 0489 | GPU device information printing | `    std::cout << "Global memory GiB: " << std::fixed << std::setprecision(2)` | Formats output so benchmark logs are easier to compare and parse. |
| 0490 | GPU device information printing | `              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << "\n";` | C++/CUDA statement within the GPU device information printing section. |
| 0491 | GPU device information printing | `    std::cout << "Warp size: " << prop.warpSize << "\n\n";` | C++/CUDA statement within the GPU device information printing section. |
| 0492 | GPU device information printing | `}` | Scope delimiter for the GPU device information printing code. |
| 0493 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0494 | usage/help text | `static void usage(const char* argv0) {` | C++/CUDA statement within the usage/help text section. |
| 0495 | usage/help text | `    std::cerr << "Usage: " << argv0 << " [all\|cpu\|vector\|stride\|occupancy\|reduction\|transpose\|bank\|layout\|copy] [N]\n";` | C++/CUDA statement within the usage/help text section. |
| 0496 | usage/help text | `}` | Scope delimiter for the usage/help text code. |
| 0497 | general C++/CUDA code | &nbsp; | Blank line separating the general C++/CUDA code code for readability. |
| 0498 | main program dispatcher | `int main(int argc, char** argv) {` | C++/CUDA statement within the main program dispatcher section. |
| 0499 | main program dispatcher | `    std::string mode = (argc > 1) ? argv[1] : "all";` | C++/CUDA statement within the main program dispatcher section. |
| 0500 | main program dispatcher | `    size_t n = (argc > 2) ? std::stoull(argv[2]) : (1ull << 24); // 16,777,216 floats` | Parses optional N argument from command line. |
| 0501 | main program dispatcher | `    int iters = 50;` | C++/CUDA statement within the main program dispatcher section. |
| 0502 | main program dispatcher | `    print_device_info();` | C++/CUDA statement within the main program dispatcher section. |
| 0503 | main program dispatcher | &nbsp; | Blank line separating the main program dispatcher code for readability. |
| 0504 | main program dispatcher | `    std::vector<Result> results;` | Allocates host-side vector storage for the main program dispatcher benchmark. |
| 0505 | main program dispatcher | &nbsp; | Blank line separating the main program dispatcher code for readability. |
| 0506 | main program dispatcher | `    if (mode == "all" \|\| mode == "cpu") {` | Dispatches benchmark cases based on command-line mode. |
| 0507 | main program dispatcher | `        std::vector<float> a(n), b(n), c(n), ref(n);` | Allocates host-side vector storage for the main program dispatcher benchmark. |
| 0508 | main program dispatcher | `        fill_vector(a); fill_vector(b);` | Initializes input data deterministically. |
| 0509 | main program dispatcher | `        results.push_back(cpu_vector_add_scalar(a, b, c, 3));` | Runs one benchmark and stores its Result. |
| 0510 | main program dispatcher | `        results.push_back(cpu_vector_add_parallel(a, b, c, 3));` | Runs one benchmark and stores its Result. |
| 0511 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0512 | main program dispatcher | `    if (mode == "all" \|\| mode == "vector") results.push_back(run_gpu_vector_add(n, iters, false));` | Dispatches benchmark cases based on command-line mode. |
| 0513 | main program dispatcher | `    if (mode == "all" \|\| mode == "stride") results.push_back(run_gpu_vector_add(n, iters, true));` | Dispatches benchmark cases based on command-line mode. |
| 0514 | main program dispatcher | `    if (mode == "all" \|\| mode == "occupancy") {` | Dispatches benchmark cases based on command-line mode. |
| 0515 | main program dispatcher | `        results.push_back(run_occupancy(n, iters, false));` | Runs one benchmark and stores its Result. |
| 0516 | main program dispatcher | `        results.push_back(run_occupancy(n, iters, true));` | Runs one benchmark and stores its Result. |
| 0517 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0518 | main program dispatcher | `    if (mode == "all" \|\| mode == "reduction") {` | Dispatches benchmark cases based on command-line mode. |
| 0519 | main program dispatcher | `        results.push_back(run_reduction(n, 10, true));` | Runs one benchmark and stores its Result. |
| 0520 | main program dispatcher | `        results.push_back(run_reduction(n, 20, false));` | Runs one benchmark and stores its Result. |
| 0521 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0522 | main program dispatcher | `    if (mode == "all" \|\| mode == "transpose") {` | Dispatches benchmark cases based on command-line mode. |
| 0523 | main program dispatcher | `        results.push_back(run_transpose(4096, 4096, 20, false));` | Runs one benchmark and stores its Result. |
| 0524 | main program dispatcher | `        results.push_back(run_transpose(4096, 4096, 20, true));` | Runs one benchmark and stores its Result. |
| 0525 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0526 | main program dispatcher | `    if (mode == "all" \|\| mode == "bank") {` | Dispatches benchmark cases based on command-line mode. |
| 0527 | main program dispatcher | `        results.push_back(run_bank_conflict(1000, false));` | Runs one benchmark and stores its Result. |
| 0528 | main program dispatcher | `        results.push_back(run_bank_conflict(1000, true));` | Runs one benchmark and stores its Result. |
| 0529 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0530 | main program dispatcher | `    if (mode == "all" \|\| mode == "layout") {` | Dispatches benchmark cases based on command-line mode. |
| 0531 | main program dispatcher | `        results.push_back(run_aos_soa(n, iters, false));` | Runs one benchmark and stores its Result. |
| 0532 | main program dispatcher | `        results.push_back(run_aos_soa(n, iters, true));` | Runs one benchmark and stores its Result. |
| 0533 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0534 | main program dispatcher | `    if (mode == "all" \|\| mode == "copy") {` | Dispatches benchmark cases based on command-line mode. |
| 0535 | main program dispatcher | `        results.push_back(run_copy(n, 30, false));` | Runs one benchmark and stores its Result. |
| 0536 | main program dispatcher | `        results.push_back(run_copy(n, 30, true));` | Runs one benchmark and stores its Result. |
| 0537 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0538 | main program dispatcher | &nbsp; | Blank line separating the main program dispatcher code for readability. |
| 0539 | main program dispatcher | `    if (results.empty()) {` | Detects invalid mode where no benchmark was selected. |
| 0540 | main program dispatcher | `        usage(argv[0]);` | C++/CUDA statement within the main program dispatcher section. |
| 0541 | main program dispatcher | `        return 2;` | Returns from the current function in the main program dispatcher section. |
| 0542 | main program dispatcher | `    }` | Scope delimiter for the main program dispatcher code. |
| 0543 | main program dispatcher | &nbsp; | Blank line separating the main program dispatcher code for readability. |
| 0544 | main program dispatcher | `    for (const auto& r : results) print_result(r);` | Defines or calls formatted benchmark result printing. |
| 0545 | main program dispatcher | `    bool ok = std::all_of(results.begin(), results.end(), [](const Result& r){ return r.ok; });` | Stores validation status. |
| 0546 | main program dispatcher | `    return ok ? 0 : 1;` | Returns from the current function in the main program dispatcher section. |
| 0547 | main program dispatcher | `}` | Scope delimiter for the main program dispatcher code. |


---

## Recommended profiling commands

```bash
mkdir -p reports

for mode in vector stride occupancy reduction transpose bank layout copy; do
  ncu --set full --target-processes all --force-overwrite \
      -o reports/ncu_${mode} \
      ./build/gpu_cpu_optimization ${mode} 16777216 \
      > reports/ncu_${mode}.txt 2>&1
done
```

For `transpose` and `bank`, the second numeric argument is ignored or not required, so you can also run:

```bash
ncu --set full --target-processes all --force-overwrite \
    -o reports/ncu_transpose \
    ./build/gpu_cpu_optimization transpose \
    > reports/ncu_transpose.txt 2>&1

ncu --set full --target-processes all --force-overwrite \
    -o reports/ncu_bank \
    ./build/gpu_cpu_optimization bank \
    > reports/ncu_bank.txt 2>&1
```

---

## One strong  summary

```text
I built this lab to practice profiler-guided optimization. For every case, I first
validate correctness, then measure baseline timing, then classify the bottleneck.
The cases cover memory bandwidth, coalescing, occupancy/register pressure,
atomic contention, warp-level reduction, shared-memory bank conflicts, data
layout, and host-device transfer overhead. The important point is that I do not
blindly chase one metric like occupancy or L2 hit rate. I interpret each metric
in the context of the workload.
```
