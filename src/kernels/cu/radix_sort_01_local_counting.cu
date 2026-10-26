#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/utils.h>
#include "../defines.h"

__global__ void radix_sort_01_local_counting(
    const unsigned* values, unsigned* counts, unsigned n, unsigned bit)
{
    const unsigned local = threadIdx.x;
    const size_t index = size_t(blockIdx.x) * GROUP_SIZE + local;
    __shared__ unsigned data[GROUP_SIZE];
    data[local] = index < n ? (values[index] >> bit) & 1u : 0;
    __syncthreads();
    for (unsigned step = GROUP_SIZE / 2; step > 0; step /= 2) {
        if (local < step) data[local] += data[local + step];
        __syncthreads();
    }
    if (local == 0) {
        counts[blockIdx.x] = data[0];
        if (blockIdx.x + 1 == gridDim.x) counts[gridDim.x] = 0;
    }
}

namespace cuda {
void radix_sort_01_local_counting(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& values, gpu::gpu_mem_32u& counts, unsigned n, unsigned bit)
{
    const cudaStream_t stream = gpu::Context().cudaStream();
    ::radix_sort_01_local_counting<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), counts.cuptr(), n, bit);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
}
