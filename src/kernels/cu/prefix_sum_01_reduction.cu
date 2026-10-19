#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/utils.h>
#include <cstdint>
#include "../defines.h"

__global__ void prefix_sum_01_reduction(const unsigned* input, unsigned* sums, unsigned n)
{
    const unsigned tid = threadIdx.x, lane = tid & 31;
    const size_t begin = size_t(blockIdx.x) * GROUP_SIZE * PREFIX_SUM_VECTORS * 4;
    unsigned sum = 0;
    #pragma unroll
    for (unsigned v = 0; v < PREFIX_SUM_VECTORS; ++v) {
        const size_t index = begin + 4 * (v * GROUP_SIZE + tid);
        if (index + 3 < n && (reinterpret_cast<uintptr_t>(input) & 15) == 0) {
            const uint4 x = *reinterpret_cast<const uint4*>(input + index);
            sum += x.x + x.y + x.z + x.w;
        } else {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                if (index + j < n) sum += input[index + j];
        }
    }
    #pragma unroll
    for (unsigned d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
    __shared__ unsigned warp_sums[GROUP_SIZE / 32];
    if (lane == 0) warp_sums[tid / 32] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = tid < GROUP_SIZE / 32 ? warp_sums[tid] : 0;
        #pragma unroll
        for (unsigned d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (tid == 0) sums[blockIdx.x] = sum;
    }
}

namespace cuda {
void prefix_sum_01_reduction(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& sums, unsigned n)
{
    if (!n) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312);
    const size_t tile = GROUP_SIZE * PREFIX_SUM_VECTORS * 4;
    const size_t blocks = (size_t(n) + tile - 1) / tile;
    const dim3 block = workSize.cuBlockSize(), grid = workSize.cuGridSize();
    rassert(block.x == GROUP_SIZE && block.y == 1 && block.z == 1
            && grid.x == blocks && grid.y == 1 && grid.z == 1, 7419301031);
    rassert(input.size() >= size_t(n) * sizeof(unsigned)
            && sums.size() >= blocks * sizeof(unsigned), 7419301032);
    const cudaStream_t stream = context.cudaStream();
    ::prefix_sum_01_reduction<<<grid, block, 0, stream>>>(input.cuptr(), sums.cuptr(), n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
