#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/utils.h>
#include "../defines.h"
#include "helpers/radix_scan.cuh"

__global__ void radix_sort_03_global_prefixes_scan_accumulation(unsigned* sums, unsigned n)
{
    __shared__ unsigned values[GROUP_SIZE];
    unsigned carry = 0;
    for (size_t begin = 0; begin < n; begin += RADIX_SCAN_TILE) {
        const size_t index = begin + threadIdx.x;
        const unsigned value = index < n ? sums[index] : 0;
        const unsigned inclusive = radix_scan::block_scan(value, values);
        const unsigned total = values[GROUP_SIZE - 1];
        __syncthreads();
        if (index < n)
            sums[index] = carry + inclusive - value;
        carry += total;
    }
}

namespace cuda {
void radix_sort_03_global_prefixes_scan_accumulation(gpu::gpu_mem_32u& sums, unsigned n)
{
    if (n == 0)
        return;
    const cudaStream_t stream = gpu::Context().cudaStream();
    ::radix_sort_03_global_prefixes_scan_accumulation<<<1, GROUP_SIZE, 0, stream>>>(
        sums.cuptr(), n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
