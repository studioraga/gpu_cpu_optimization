#  Topic to Code Mapping

## CV-fit questions

Use `copy`, `cpu`, `vector`, and `layout` cases to explain full-stack performance reasoning. The same discipline used in SoC/BSP/memory/camera work applies to GPU optimization: observe data movement, classify bottleneck, fix layout/copy/compute, and validate.

## Vector add

Run:

```bash
build/gpu_cpu_optimization vector
```

Explain: vector add is memory-bandwidth-bound because it performs two loads, one store, and one add per element. L2 hit rate is not a decisive metric because the data stream has little reuse.

## L2 hit rate useful vs misleading

Useful: transpose, layout, tiled algorithms, stencils, repeated lookup. Misleading: streaming copy, vector add, one-pass transforms.

## Occupancy

Run:

```bash
build/gpu_cpu_optimization occupancy
```

Explain: low occupancy may come from register pressure; high occupancy does not guarantee throughput. Check achieved occupancy together with stall reasons and memory/compute throughput.

## Reduction

Run:

```bash
build/gpu_cpu_optimization reduction
```

Explain: atomic reduction is simple but can serialize updates. Block/warp reduction reduces atomics and synchronization pressure.

## Transpose

Run:

```bash
build/gpu_cpu_optimization transpose
```

Explain: naive transpose has uncoalesced global stores. Tiled transpose uses shared memory to make global access coalesced and uses padding to avoid shared-memory bank conflicts.

## Strided memory / coalescing

Run:

```bash
build/gpu_cpu_optimization stride
```

Explain: adjacent lanes access far-apart addresses; this increases memory transactions and reduces useful bandwidth.

## C++17 resource safety

See `include/cuda_check.hpp`: DeviceBuffer and PinnedHostBuffer use RAII, delete copy, and free resources safely.

## Python/Git workflow

Use scripts for repeatable profile collection. Keep each optimization in a small commit: baseline, profile, change, validation, documentation.
