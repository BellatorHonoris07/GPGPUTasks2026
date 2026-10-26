#pragma once

#include <cuda_runtime.h>
#include "../../defines.h"

namespace radix_scan {

__device__ inline unsigned block_scan(unsigned value, unsigned* shared)
{
    const unsigned tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (unsigned offset = 1; offset < GROUP_SIZE; offset *= 2) {
        const unsigned previous = tid >= offset ? shared[tid - offset] : 0;
        __syncthreads();
        shared[tid] += previous;
        __syncthreads();
    }
    return shared[tid];
}

} // namespace radix_scan
