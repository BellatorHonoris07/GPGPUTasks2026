#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/cuda/utils.h>
#include "../defines.h"
#include "helpers/radix_scan.cuh"

__global__ void radix_sort_02_global_prefixes_scan_sum_reduction(
    unsigned* histogram, unsigned* sums, unsigned n)
{
    __shared__ unsigned values[GROUP_SIZE];
    const size_t index = size_t(blockIdx.x) * RADIX_SCAN_TILE + threadIdx.x;
    const unsigned value = index < n ? histogram[index] : 0;
    const unsigned inclusive = radix_scan::block_scan(value, values);
    if (index < n)
        histogram[index] = inclusive - value;
    if (threadIdx.x == 0)
        sums[blockIdx.x] = values[GROUP_SIZE - 1];
}

namespace cuda {
void radix_sort_02_global_prefixes_scan_sum_reduction(const gpu::WorkSize& workSize,
    gpu::gpu_mem_32u& histogram, gpu::gpu_mem_32u& sums, unsigned n)
{
    if (n == 0)
        return;
    const cudaStream_t stream = gpu::Context().cudaStream();
    ::radix_sort_02_global_prefixes_scan_sum_reduction<<<
        workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        histogram.cuptr(), sums.cuptr(), n);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}
} // namespace cuda
