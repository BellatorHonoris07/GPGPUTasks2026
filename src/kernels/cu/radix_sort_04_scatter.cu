#include <libgpu/context.h>
#include <libgpu/cuda/utils.h>
#include <cstdint>
#include "../defines.h"
#include "../kernels.h"
#include "helpers/radix_scan.cuh"

__global__ void radix_sort_04_scatter(const unsigned* values, const unsigned* prefixes,
    const unsigned* sums, unsigned* output, unsigned n, unsigned bit)
{
    const size_t index = size_t(blockIdx.x) * GROUP_SIZE + threadIdx.x;
    const unsigned value = index < n ? values[index] : 0;
    const unsigned one = index < n ? (value >> bit) & 1u : 0;
    __shared__ unsigned data[GROUP_SIZE];
    unsigned ones = radix_scan::block_scan(one, data);
    if (index < n) {
        ones += prefixes[blockIdx.x] + sums[blockIdx.x / GROUP_SIZE];
        const unsigned totalOnes = prefixes[gridDim.x] + sums[gridDim.x / GROUP_SIZE];
        const unsigned position = one ? n - totalOnes + ones - 1 : index - ones;
        output[position] = value;
    }
}

namespace cuda {
void radix_sort_04_scatter(const gpu::WorkSize& workSize, const gpu::gpu_mem_32u& values,
    const gpu::gpu_mem_32u& prefixes, const gpu::gpu_mem_32u& sums,
    gpu::gpu_mem_32u& output, unsigned n, unsigned bit)
{
    const cudaStream_t stream = gpu::Context().cudaStream();
    ::radix_sort_04_scatter<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(
        values.cuptr(), prefixes.cuptr(), sums.cuptr(), output.cuptr(), n, bit);
    CUDA_CHECK_KERNEL_ASYNC(stream);
}

void radix_sort(const gpu::gpu_mem_32u& input, gpu::gpu_mem_32u& output, unsigned n)
{
    if (!n) return;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 7419501001);
    rassert(input.size() >= size_t(n) * sizeof(unsigned)
            && output.size() >= size_t(n) * sizeof(unsigned), 7419501002);
    const auto in = reinterpret_cast<std::uintptr_t>(input.cuptr());
    const auto out = reinterpret_cast<std::uintptr_t>(output.cuptr());
    rassert(in + size_t(n) * sizeof(unsigned) <= out
            || out + size_t(n) * sizeof(unsigned) <= in, 7419501003);
    const unsigned blocks = (size_t(n) + GROUP_SIZE - 1) / GROUP_SIZE;
    const unsigned chunks = (blocks + 1 + GROUP_SIZE - 1) / GROUP_SIZE;
    gpu::gpu_mem_32u temporary(n), prefixes(blocks + 1), sums(chunks);
    const gpu::WorkSize tiles(GROUP_SIZE, size_t(blocks) * GROUP_SIZE);
    const gpu::WorkSize scan(GROUP_SIZE, size_t(chunks) * GROUP_SIZE);
    const gpu::gpu_mem_32u* source = &input;
    for (unsigned bit = 0; bit < 32; ++bit) {
        auto& destination = bit % 2 == 0 ? temporary : output;
        radix_sort_01_local_counting(tiles, *source, prefixes, n, bit);
        radix_sort_02_global_prefixes_scan_sum_reduction(scan, prefixes, sums, blocks + 1);
        radix_sort_03_global_prefixes_scan_accumulation(sums, chunks);
        radix_sort_04_scatter(tiles, *source, prefixes, sums, destination, n, bit);
        source = &destination;
    }
    CUDA_CHECK_KERNEL_SYNC(context.cudaStream());
}
}
