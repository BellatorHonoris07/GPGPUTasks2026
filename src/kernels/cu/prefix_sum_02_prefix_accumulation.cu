#include <libgpu/context.h>
#include <libgpu/device_memory_pool.h>
#include <libgpu/cuda/utils.h>
#include <cstddef>
#include <cstdint>
#include "../defines.h"
#include "../kernels.h"

namespace prefix_scan {
constexpr unsigned threads = GROUP_SIZE, vectors = PREFIX_SUM_VECTORS, warps = threads / 32;
constexpr unsigned tile_size = threads * vectors * 4;
static_assert(threads % 32 == 0 && warps * vectors == 32);
struct PeerSums { const unsigned* values[4] = {}; };
__global__ void sum_peers(PeerSums peers, unsigned* offset) {
    unsigned sum = 0;
    for (const unsigned* value : peers.values) if (value) sum += *value;
    *offset = sum;
}
__global__ void add_offset(unsigned* output, unsigned n, const unsigned* offset) {
    const size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) output[index] += *offset;
}
__device__ unsigned warp_scan(unsigned value) {
    #pragma unroll
    for (unsigned d = 1; d < 32; d *= 2) {
        unsigned previous = __shfl_up_sync(0xffffffffu, value, d);
        if ((threadIdx.x & 31) >= d) value += previous;
    }
    return value;
}
__global__ void scan_tiles(const unsigned* input, unsigned* output, unsigned n,
                       const unsigned* offsets, unsigned initial, unsigned* total) {
    __shared__ unsigned totals[32];
    const unsigned tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    unsigned carry = initial;
    for (size_t begin = size_t(blockIdx.x) * tile_size; begin < n;
         begin += size_t(gridDim.x) * tile_size) {
        const bool full = size_t(n) - begin >= tile_size
            && ((reinterpret_cast<uintptr_t>(input) | reinterpret_cast<uintptr_t>(output)) & 15) == 0;
        uint4 values[vectors];
        #pragma unroll
        for (unsigned v = 0; v < vectors; ++v) {
            const size_t index = begin + 4 * (v * threads + tid);
            uint4 x{};
            if (full) x = *reinterpret_cast<const uint4*>(input + index);
            else {
                if (index < n) x.x = input[index];
                if (index + 1 < n) x.y = input[index + 1];
                if (index + 2 < n) x.z = input[index + 2];
                if (index + 3 < n) x.w = input[index + 3];
            }
            x.y += x.x; x.z += x.y; x.w += x.z;
            unsigned prefix = warp_scan(x.w), before = prefix - x.w;
            x.x += before; x.y += before; x.z += before; x.w += before;
            values[v] = x;
            if (lane == 31) totals[v * warps + warp] = prefix;
        }
        __syncthreads();
        if (warp == 0) totals[lane] = warp_scan(totals[lane]);
        __syncthreads();
        unsigned base = carry + (offsets && blockIdx.x ? offsets[blockIdx.x - 1] : 0);
        #pragma unroll
        for (unsigned v = 0; v < vectors; ++v) {
            unsigned segment = v * warps + warp;
            unsigned add = base + (segment ? totals[segment - 1] : 0);
            uint4 x = values[v];
            x.x += add; x.y += add; x.z += add; x.w += add;
            const size_t index = begin + 4 * (v * threads + tid);
            if (full) *reinterpret_cast<uint4*>(output + index) = x;
            else {
                if (index < n) output[index] = x.x;
                if (index + 1 < n) output[index + 1] = x.y;
                if (index + 2 < n) output[index + 2] = x.z;
                if (index + 3 < n) output[index + 3] = x.w;
            }
        }
        if (total && tid == 0 && blockIdx.x + 1 == gridDim.x) *total = base + totals[31];
        carry += totals[31];
        __syncthreads();
    }
}
} // namespace prefix_scan

namespace cuda {
void prefix_sum_add_offsets(gpu::gpu_mem_32u& output, unsigned n,
                            const unsigned* const* totals, unsigned count) {
    if (!n || !count) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA && output.size() >= size_t(n) * sizeof(unsigned), 7419301041);
    rassert(totals && count <= 4, 7419301044);
    prefix_scan::PeerSums peers;
    for (unsigned i = 0; i < count; ++i) { rassert(totals[i], 7419301045); peers.values[i] = totals[i]; }
    auto offset = context.memoryPool().allocateN<unsigned>("prefix_sum.offset", 1);
    const cudaStream_t stream = context.cudaStream();
    prefix_scan::sum_peers<<<1, 1, 0, stream>>>(peers, offset.cuptr());
    prefix_scan::add_offset<<<(size_t(n) + 255) / 256, 256, 0, stream>>>(output.cuptr(), n, offset.cuptr());
    CUDA_CHECK_KERNEL_SYNC(stream);
}
void prefix_sum(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output,
                unsigned int n, unsigned int initial, unsigned* total) {
    if (!n) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 7419301021);
    rassert(input.size() >= size_t(n) * sizeof(unsigned)
            && output.size() >= size_t(n) * sizeof(unsigned), 7419301022);
    const auto in = reinterpret_cast<std::uintptr_t>(input.cuptr());
    const auto out = reinterpret_cast<std::uintptr_t>(output.cuptr());
    rassert(in + size_t(n) * sizeof(unsigned) <= out
            || out + size_t(n) * sizeof(unsigned) <= in, 7419301023);
    const unsigned blocks = (size_t(n) + prefix_scan::tile_size - 1) / prefix_scan::tile_size;
    auto sums = context.memoryPool().allocateN<unsigned>("prefix_sum.sums", blocks);
    const cudaStream_t stream = context.cudaStream();
    using prefix_scan::scan_tiles;
    if (blocks > 1) {
        prefix_sum_01_reduction(gpu::WorkSize(prefix_scan::threads,
            size_t(blocks) * prefix_scan::threads), input, sums, n);
        scan_tiles<<<1, prefix_scan::threads, 0, stream>>>(
            sums.cuptr(), sums.cuptr(), blocks, nullptr, 0, nullptr);
    }
    scan_tiles<<<blocks, prefix_scan::threads, 0, stream>>>(
        input.cuptr(), output.cuptr(), n, blocks > 1 ? sums.cuptr() : nullptr, initial, total);
    CUDA_CHECK_KERNEL_SYNC(stream);
}
} // namespace cuda
