#include "cuda_check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

struct Result {
    std::string name;
    float ms;
    double gbps;
    double gops;
    bool ok;
};

static double now_ms_cpu() {
    using clock = std::chrono::high_resolution_clock;
    static const auto t0 = clock::now();
    auto t = clock::now();
    return std::chrono::duration<double, std::milli>(t - t0).count();
}

static void fill_vector(std::vector<float>& v) {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

static bool approx_equal(const std::vector<float>& a, const std::vector<float>& b, float tol = 1e-3f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) {
            std::cerr << "Mismatch at " << i << ": " << a[i] << " vs " << b[i] << "\n";
            return false;
        }
    }
    return true;
}

static void print_result(const Result& r) {
    std::cout << std::left << std::setw(34) << r.name
              << " time_ms=" << std::setw(10) << std::fixed << std::setprecision(4) << r.ms
              << " GB/s=" << std::setw(10) << std::setprecision(2) << r.gbps
              << " GOP/s=" << std::setw(10) << std::setprecision(2) << r.gops
              << " status=" << (r.ok ? "PASS" : "FAIL") << "\n";
}

// -------------------------------------------------------------------------------------------------
// 1. CPU baseline: scalar and optional OpenMP.  point: CPU memory bandwidth is also a limit.
// -------------------------------------------------------------------------------------------------
static Result cpu_vector_add_scalar(const std::vector<float>& a, const std::vector<float>& b,
                                    std::vector<float>& c, int iters) {
    double t0 = now_ms_cpu();
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < a.size(); ++i) c[i] = a[i] + b[i];
    }
    double t1 = now_ms_cpu();
    double ms = (t1 - t0) / iters;
    double bytes = 3.0 * a.size() * sizeof(float);
    return {"cpu_vector_add_scalar", static_cast<float>(ms), bytes / (ms * 1.0e6), a.size() / (ms * 1.0e6), true};
}

static Result cpu_vector_add_parallel(const std::vector<float>& a, const std::vector<float>& b,
                                      std::vector<float>& c, int iters) {
    double t0 = now_ms_cpu();
    for (int it = 0; it < iters; ++it) {
        #ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (long long i = 0; i < static_cast<long long>(a.size()); ++i) c[i] = a[i] + b[i];
    }
    double t1 = now_ms_cpu();
    double ms = (t1 - t0) / iters;
    double bytes = 3.0 * a.size() * sizeof(float);
    return {"cpu_vector_add_parallel", static_cast<float>(ms), bytes / (ms * 1.0e6), a.size() / (ms * 1.0e6), true};
}

// -------------------------------------------------------------------------------------------------
// 2. GPU vector add: coalesced streaming kernel.  point: memory-bound; L2 hit rate may mislead.
// -------------------------------------------------------------------------------------------------
__global__ void vector_add_coalesced_kernel(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float* __restrict__ c,
                                            size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) c[i] = a[i] + b[i];
}

// -------------------------------------------------------------------------------------------------
// 3. Strided/uncoalesced access: adjacent warp lanes touch far-apart memory.
//  point: high transaction count and poor memory efficiency.
// -------------------------------------------------------------------------------------------------
__global__ void vector_add_strided_kernel(const float* __restrict__ a,
                                          const float* __restrict__ b,
                                          float* __restrict__ c,
                                          size_t logical_n,
                                          int stride_elems) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_stride = blockDim.x * gridDim.x;
    for (; i < logical_n; i += grid_stride) {
        size_t j = i * static_cast<size_t>(stride_elems);
        c[j] = a[j] + b[j];
    }
}

static Result run_gpu_vector_add(size_t n, int iters, bool strided) {
    std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
    fill_vector(h_a); fill_vector(h_b);
    for (size_t i = 0; i < n; ++i) h_ref[i] = h_a[i] + h_b[i];

    DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), d_b.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_c.get(), 0, d_c.bytes()));

    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));
    GpuTimer timer;

    if (!strided) {
        vector_add_coalesced_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) vector_add_coalesced_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), n);
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), d_c.bytes(), cudaMemcpyDeviceToHost));
        bool ok = approx_equal(h_c, h_ref);
        double bytes = 3.0 * n * sizeof(float);
        return {"gpu_vector_add_coalesced", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
    } else {
        int stride = 8;
        size_t logical_n = n / stride;
        vector_add_strided_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), logical_n, stride);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) vector_add_strided_kernel<<<grid, block>>>(d_a.get(), d_b.get(), d_c.get(), logical_n, stride);
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), d_c.bytes(), cudaMemcpyDeviceToHost));
        bool ok = true;
        for (size_t i = 0; i < logical_n; ++i) {
            size_t j = i * stride;
            if (std::fabs(h_c[j] - h_ref[j]) > 1e-3f) { ok = false; break; }
        }
        // Requested useful bytes, not total address span. This exposes efficiency loss.
        double bytes = 3.0 * logical_n * sizeof(float);
        return {"gpu_vector_add_strided", ms, bytes / (ms * 1.0e6), logical_n / (ms * 1.0e6), ok};
    }
}

// -------------------------------------------------------------------------------------------------
// 4. Occupancy/register pressure demo.  point: occupancy is not the final performance metric.
// -------------------------------------------------------------------------------------------------
__global__ void low_register_kernel(const float* __restrict__ a, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float x = a[i];
        #pragma unroll 8
        for (int k = 0; k < 32; ++k) x = x * 1.000001f + 0.000001f;
        out[i] = x;
    }
}

__global__ void high_register_kernel(const float* __restrict__ a, float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        float x = a[i];
        float r0=x, r1=x+1, r2=x+2, r3=x+3, r4=x+4, r5=x+5, r6=x+6, r7=x+7;
        float r8=x+8, r9=x+9, r10=x+10, r11=x+11, r12=x+12, r13=x+13, r14=x+14, r15=x+15;
        #pragma unroll 8
        for (int k = 0; k < 64; ++k) {
            r0 = r0 * 1.000001f + r8;   r1 = r1 * 0.999999f + r9;
            r2 = r2 * 1.000002f + r10;  r3 = r3 * 0.999998f + r11;
            r4 = r4 * 1.000003f + r12;  r5 = r5 * 0.999997f + r13;
            r6 = r6 * 1.000004f + r14;  r7 = r7 * 0.999996f + r15;
            r8 += 0.000001f; r9 += 0.000002f; r10 += 0.000003f; r11 += 0.000004f;
            r12 += 0.000005f; r13 += 0.000006f; r14 += 0.000007f; r15 += 0.000008f;
        }
        out[i] = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15;
    }
}

static Result run_occupancy(size_t n, int iters, bool high) {
    std::vector<float> h_a(n), h_out(n);
    fill_vector(h_a);
    DeviceBuffer<float> d_a(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), d_a.bytes(), cudaMemcpyHostToDevice));
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));
    GpuTimer timer;
    if (high) high_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);
    else low_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);
    CUDA_KERNEL_CHECK();
    timer.start();
    for (int it = 0; it < iters; ++it) {
        if (high) high_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);
        else low_register_kernel<<<grid, block>>>(d_a.get(), d_out.get(), n);
    }
    float ms = timer.stop_ms() / iters;
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    bool ok = std::isfinite(h_out[n / 2]);
    return {high ? "high_register_occupancy_demo" : "low_register_occupancy_demo", ms, 0.0, n / (ms * 1.0e6), ok};
}

// -------------------------------------------------------------------------------------------------
// 5. Reduction: bad atomic vs block/warp reduction.  point: atomics/sync/staging matter.
// -------------------------------------------------------------------------------------------------
__global__ void reduce_atomic_bad_kernel(const float* __restrict__ in, float* out, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (; i < n; i += stride) sum += in[i];
    atomicAdd(out, sum);
}

__inline__ __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__global__ void reduce_block_warp_kernel(const float* __restrict__ in, float* partial, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (; i < n; i += stride) sum += in[i];

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) partial[blockIdx.x] = block_sum;
    }
}

static Result run_reduction(size_t n, int iters, bool atomic_bad) {
    std::vector<float> h_in(n);
    fill_vector(h_in);
    double ref = std::accumulate(h_in.begin(), h_in.end(), 0.0);
    DeviceBuffer<float> d_in(n);
    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), d_in.bytes(), cudaMemcpyHostToDevice));
    int block = 256;
    int grid = static_cast<int>(std::min<size_t>((n + block - 1) / block, 4096));
    GpuTimer timer;
    float sum = 0.0f;

    if (atomic_bad) {
        DeviceBuffer<float> d_out(1);
        CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(float)));
        reduce_atomic_bad_kernel<<<grid, block>>>(d_in.get(), d_out.get(), n);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) {
            CUDA_CHECK(cudaMemset(d_out.get(), 0, sizeof(float)));
            reduce_atomic_bad_kernel<<<grid, block>>>(d_in.get(), d_out.get(), n);
        }
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(&sum, d_out.get(), sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = std::fabs(sum - static_cast<float>(ref)) / std::max(1.0f, std::fabs(static_cast<float>(ref))) < 1e-2f;
        double bytes = n * sizeof(float);
        return {"reduce_atomic_bad", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
    } else {
        DeviceBuffer<float> d_partial(grid);
        std::vector<float> h_partial(grid);
        reduce_block_warp_kernel<<<grid, block>>>(d_in.get(), d_partial.get(), n);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) reduce_block_warp_kernel<<<grid, block>>>(d_in.get(), d_partial.get(), n);
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(h_partial.data(), d_partial.get(), grid * sizeof(float), cudaMemcpyDeviceToHost));
        double got = std::accumulate(h_partial.begin(), h_partial.end(), 0.0);
        bool ok = std::fabs(got - ref) / std::max(1.0, std::fabs(ref)) < 1e-2;
        double bytes = n * sizeof(float) + grid * sizeof(float);
        return {"reduce_block_warp", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
    }
}

// -------------------------------------------------------------------------------------------------
// 6. Matrix transpose: naive vs shared-memory tiled+padded.  point: coalescing and bank conflicts.
// -------------------------------------------------------------------------------------------------
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) out[x * height + (y + j)] = in[(y + j) * width + x];
    }
}

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids shared-memory bank conflicts on column reads

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) out[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

static Result run_transpose(int width, int height, int iters, bool tiled) {
    size_t n = static_cast<size_t>(width) * height;
    std::vector<float> h_in(n), h_out(n), h_ref(n);
    fill_vector(h_in);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) h_ref[x * height + y] = h_in[y * width + x];
    DeviceBuffer<float> d_in(n), d_out(n);
    CUDA_CHECK(cudaMemcpy(d_in.get(), h_in.data(), d_in.bytes(), cudaMemcpyHostToDevice));

    constexpr int TILE = 32;
    constexpr int ROWS = 8;
    dim3 block(TILE, ROWS);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    GpuTimer timer;
    if (tiled) transpose_tiled_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);
    else transpose_naive_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);
    CUDA_KERNEL_CHECK();
    timer.start();
    for (int it = 0; it < iters; ++it) {
        if (tiled) transpose_tiled_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);
        else transpose_naive_kernel<TILE, ROWS><<<grid, block>>>(d_in.get(), d_out.get(), width, height);
    }
    float ms = timer.stop_ms() / iters;
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    bool ok = approx_equal(h_out, h_ref);
    double bytes = 2.0 * n * sizeof(float);
    return {tiled ? "transpose_tiled_padded" : "transpose_naive", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
}

// -------------------------------------------------------------------------------------------------
// 7. Shared memory bank-conflict micro demo.  point: padding changes bank mapping.
// -------------------------------------------------------------------------------------------------
__global__ void shared_bank_conflict_kernel(float* out, int reps) {
    __shared__ float tile[32][32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float v = static_cast<float>(tx + ty);
    tile[ty][tx] = v;
    __syncthreads();
    float sum = 0.0f;
    for (int r = 0; r < reps; ++r) sum += tile[tx][ty]; // column-style read can conflict
    out[ty * 32 + tx] = sum;
}

__global__ void shared_bank_padded_kernel(float* out, int reps) {
    __shared__ float tile[32][33];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float v = static_cast<float>(tx + ty);
    tile[ty][tx] = v;
    __syncthreads();
    float sum = 0.0f;
    for (int r = 0; r < reps; ++r) sum += tile[tx][ty];
    out[ty * 32 + tx] = sum;
}

static Result run_bank_conflict(int iters, bool padded) {
    DeviceBuffer<float> d_out(32 * 32);
    dim3 block(32, 32);
    int reps = 512;
    if (padded) shared_bank_padded_kernel<<<1, block>>>(d_out.get(), reps);
    else shared_bank_conflict_kernel<<<1, block>>>(d_out.get(), reps);
    CUDA_KERNEL_CHECK();
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < iters; ++it) {
        if (padded) shared_bank_padded_kernel<<<1, block>>>(d_out.get(), reps);
        else shared_bank_conflict_kernel<<<1, block>>>(d_out.get(), reps);
    }
    float ms = timer.stop_ms() / iters;
    std::vector<float> h(32 * 32);
    CUDA_CHECK(cudaMemcpy(h.data(), d_out.get(), d_out.bytes(), cudaMemcpyDeviceToHost));
    bool ok = std::isfinite(h[17]);
    return {padded ? "shared_bank_padded" : "shared_bank_conflict", ms, 0.0, 0.0, ok};
}

// -------------------------------------------------------------------------------------------------
// 8. AoS vs SoA data layout.  point: memory layout can dominate simple arithmetic.
// -------------------------------------------------------------------------------------------------
struct ParticleAoS { float x, y, z, vx, vy, vz; };

__global__ void update_aos_kernel(ParticleAoS* p, float dt, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i].x += p[i].vx * dt;
}

__global__ void update_soa_kernel(float* x, const float* vx, float dt, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += vx[i] * dt;
}

static Result run_aos_soa(size_t n, int iters, bool soa) {
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    GpuTimer timer;
    bool ok = true;
    if (!soa) {
        std::vector<ParticleAoS> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = {1,2,3,0.5f,0.6f,0.7f};
        DeviceBuffer<ParticleAoS> d(n);
        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));
        update_aos_kernel<<<grid, block>>>(d.get(), 0.1f, n);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) update_aos_kernel<<<grid, block>>>(d.get(), 0.1f, n);
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(h.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost));
        ok = std::isfinite(h[n/2].x);
        double useful_bytes = 2.0 * n * sizeof(float) + n * sizeof(float);
        return {"aos_update_x", ms, useful_bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
    } else {
        std::vector<float> h_x(n, 1.0f), h_vx(n, 0.5f);
        DeviceBuffer<float> d_x(n), d_vx(n);
        CUDA_CHECK(cudaMemcpy(d_x.get(), h_x.data(), d_x.bytes(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx.get(), h_vx.data(), d_vx.bytes(), cudaMemcpyHostToDevice));
        update_soa_kernel<<<grid, block>>>(d_x.get(), d_vx.get(), 0.1f, n);
        CUDA_KERNEL_CHECK();
        timer.start();
        for (int it = 0; it < iters; ++it) update_soa_kernel<<<grid, block>>>(d_x.get(), d_vx.get(), 0.1f, n);
        float ms = timer.stop_ms() / iters;
        CUDA_CHECK(cudaMemcpy(h_x.data(), d_x.get(), d_x.bytes(), cudaMemcpyDeviceToHost));
        ok = std::isfinite(h_x[n/2]);
        double bytes = 3.0 * n * sizeof(float);
        return {"soa_update_x", ms, bytes / (ms * 1.0e6), n / (ms * 1.0e6), ok};
    }
}

// -------------------------------------------------------------------------------------------------
// 9. Host/device copy: pageable vs pinned.  point: end-to-end pipeline may bottleneck outside kernel.
// -------------------------------------------------------------------------------------------------
static Result run_copy(size_t n, int iters, bool pinned) {
    DeviceBuffer<float> d(n);
    GpuTimer timer;
    float ms = 0.0f;
    if (!pinned) {
        std::vector<float> h(n, 1.0f);
        CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));
        timer.start();
        for (int it = 0; it < iters; ++it) CUDA_CHECK(cudaMemcpy(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice));
        ms = timer.stop_ms() / iters;
    } else {
        PinnedHostBuffer<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = 1.0f;
        CUDA_CHECK(cudaMemcpy(d.get(), h.get(), d.bytes(), cudaMemcpyHostToDevice));
        timer.start();
        for (int it = 0; it < iters; ++it) CUDA_CHECK(cudaMemcpy(d.get(), h.get(), d.bytes(), cudaMemcpyHostToDevice));
        ms = timer.stop_ms() / iters;
    }
    double bytes = n * sizeof(float);
    return {pinned ? "h2d_copy_pinned" : "h2d_copy_pageable", ms, bytes / (ms * 1.0e6), 0.0, true};
}

static void print_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Global memory GiB: " << std::fixed << std::setprecision(2)
              << static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0) << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n\n";
}

static void usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [all|cpu|vector|stride|occupancy|reduction|transpose|bank|layout|copy] [N]\n";
}

int main(int argc, char** argv) {
    std::string mode = (argc > 1) ? argv[1] : "all";
    size_t n = (argc > 2) ? std::stoull(argv[2]) : (1ull << 24); // 16,777,216 floats
    int iters = 50;
    print_device_info();

    std::vector<Result> results;

    if (mode == "all" || mode == "cpu") {
        std::vector<float> a(n), b(n), c(n), ref(n);
        fill_vector(a); fill_vector(b);
        results.push_back(cpu_vector_add_scalar(a, b, c, 3));
        results.push_back(cpu_vector_add_parallel(a, b, c, 3));
    }
    if (mode == "all" || mode == "vector") results.push_back(run_gpu_vector_add(n, iters, false));
    if (mode == "all" || mode == "stride") results.push_back(run_gpu_vector_add(n, iters, true));
    if (mode == "all" || mode == "occupancy") {
        results.push_back(run_occupancy(n, iters, false));
        results.push_back(run_occupancy(n, iters, true));
    }
    if (mode == "all" || mode == "reduction") {
        results.push_back(run_reduction(n, 10, true));
        results.push_back(run_reduction(n, 20, false));
    }
    if (mode == "all" || mode == "transpose") {
        results.push_back(run_transpose(4096, 4096, 20, false));
        results.push_back(run_transpose(4096, 4096, 20, true));
    }
    if (mode == "all" || mode == "bank") {
        results.push_back(run_bank_conflict(1000, false));
        results.push_back(run_bank_conflict(1000, true));
    }
    if (mode == "all" || mode == "layout") {
        results.push_back(run_aos_soa(n, iters, false));
        results.push_back(run_aos_soa(n, iters, true));
    }
    if (mode == "all" || mode == "copy") {
        results.push_back(run_copy(n, 30, false));
        results.push_back(run_copy(n, 30, true));
    }

    if (results.empty()) {
        usage(argv[0]);
        return 2;
    }

    for (const auto& r : results) print_result(r);
    bool ok = std::all_of(results.begin(), results.end(), [](const Result& r){ return r.ok; });
    return ok ? 0 : 1;
}
