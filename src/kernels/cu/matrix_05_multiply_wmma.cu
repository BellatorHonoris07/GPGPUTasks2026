#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/device_memory_pool.h>
#include <libgpu/cuda/cu/common.cu>
#include "helpers/rassert.cu"
#include "../defines.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
namespace wmma = nvcuda::wmma;
constexpr unsigned int WMMA_BM = 128, WMMA_BN = 64, WMMA_BK = 32;
constexpr unsigned int WMMA_WM = 64, WMMA_WN = 32, WMMA_THREADS = 128;
constexpr unsigned int WMMA_AP = WMMA_BM + 16, WMMA_BP = WMMA_BN + 16;
constexpr unsigned int WMMA_AS = WMMA_BK * WMMA_AP, WMMA_BS = WMMA_BK * WMMA_BP;

template<bool TRANSPOSE>
__global__ void wmma_convert_inputs(const float* src, half* dst,
                                   unsigned rows, unsigned cols, unsigned paddedRows, unsigned paddedCols)
{
    unsigned row = blockIdx.y * 32 + threadIdx.y;
    unsigned col = blockIdx.x * 32 + threadIdx.x * 4;
    float4 value = {};
    if (row < rows && col < cols) {
        const float* input = src + size_t(row) * cols + col;
        if (col + 3 < cols && reinterpret_cast<std::uintptr_t>(input) % 16 == 0) {
            value = *reinterpret_cast<const float4*>(input);
        } else {
            value.x = input[0];
            if (col + 1 < cols) value.y = input[1];
            if (col + 2 < cols) value.z = input[2];
            if (col + 3 < cols) value.w = input[3];
        }
    }
    if constexpr (TRANSPOSE) {
        __shared__ float tile[32][33];
        const unsigned x = threadIdx.x * 4, y = threadIdx.y;
        tile[y][x] = value.x;
        tile[y][x + 1] = value.y;
        tile[y][x + 2] = value.z;
        tile[y][x + 3] = value.w;
        __syncthreads();
        value = make_float4(tile[x][y], tile[x + 1][y], tile[x + 2][y], tile[x + 3][y]);
        row = blockIdx.x * 32 + y;
        col = blockIdx.y * 32 + x;
    }
    const size_t stride = TRANSPOSE ? paddedRows : paddedCols;
    half2* packed = reinterpret_cast<half2*>(dst + size_t(row) * stride + col);
    packed[0] = __floats2half2_rn(value.x, value.y);
    packed[1] = __floats2half2_rn(value.z, value.w);
}

__global__ __launch_bounds__(WMMA_THREADS, 3)
void matrix_multiply_wmma_pipeline(const half* a, const half* b, float* __restrict__ c,
                                   unsigned int w, unsigned int h, unsigned int k)
{
    __shared__ __align__(32) half aBase[2 * WMMA_AS], bBase[2 * WMMA_BS];
    constexpr unsigned int LOAD_A = WMMA_BM * WMMA_BK / (WMMA_THREADS * 8);
    constexpr unsigned int LOAD_B = WMMA_BN * WMMA_BK / (WMMA_THREADS * 8);
    constexpr unsigned int LOADS = LOAD_A > LOAD_B ? LOAD_A : LOAD_B;
    int4 nextA[LOAD_A], nextB[LOAD_B];
    const unsigned int tid = threadIdx.x, warp = tid / 32;
    const unsigned int blockY = blockIdx.x * WMMA_BM, blockX = blockIdx.y * WMMA_BN;
    const unsigned int warpY = warp / (WMMA_BN / WMMA_WN);
    const unsigned int warpX = warp % (WMMA_BN / WMMA_WN);

    auto prefetch = [&](unsigned int baseK) {
        #pragma unroll
        for (unsigned int p = 0; p < LOADS; ++p) {
            const unsigned int index = (tid + p * WMMA_THREADS) * 8;
            if (p < LOAD_A) nextA[p] = *reinterpret_cast<const int4*>(
                a + size_t(baseK + index / WMMA_BM) * h + blockY + index % WMMA_BM);
            if (p < LOAD_B) nextB[p] = *reinterpret_cast<const int4*>(
                b + size_t(baseK + index / WMMA_BN) * w + blockX + index % WMMA_BN);
        }
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WMMA_WM / 16][WMMA_WN / 16];
    #pragma unroll
    for (unsigned int i = 0; i < WMMA_WM / 16; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < WMMA_WN / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

    prefetch(0);
    unsigned int stage = 0;
    for (unsigned int baseK = 0; baseK < k; baseK += WMMA_BK) {
        half* tileA = aBase + stage * WMMA_AS;
        half* tileB = bBase + stage * WMMA_BS;
        #pragma unroll
        for (unsigned int p = 0; p < LOADS; ++p) {
            const unsigned int index = (tid + p * WMMA_THREADS) * 8;
            if (p < LOAD_A) *reinterpret_cast<int4*>(
                tileA + (index / WMMA_BM) * WMMA_AP + index % WMMA_BM) = nextA[p];
            if (p < LOAD_B) *reinterpret_cast<int4*>(
                tileB + (index / WMMA_BN) * WMMA_BP + index % WMMA_BN) = nextB[p];
        }
        __syncthreads();
        if (baseK + WMMA_BK < k) prefetch(baseK + WMMA_BK);

        #pragma unroll
        for (unsigned int q = 0; q < WMMA_BK; q += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> ar[WMMA_WM / 16];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> br[WMMA_WN / 16];
            #pragma unroll
            for (unsigned int i = 0; i < WMMA_WM / 16; ++i)
                wmma::load_matrix_sync(ar[i], tileA + q * WMMA_AP + warpY * WMMA_WM + i * 16, WMMA_AP);
            #pragma unroll
            for (unsigned int j = 0; j < WMMA_WN / 16; ++j)
                wmma::load_matrix_sync(br[j], tileB + q * WMMA_BP + warpX * WMMA_WN + j * 16, WMMA_BP);
            #pragma unroll
            for (unsigned int i = 0; i < WMMA_WM / 16; ++i) {
                #pragma unroll
                for (unsigned int j = 0; j < WMMA_WN / 16; ++j)
                    wmma::mma_sync(acc[i][j], ar[i], br[j], acc[i][j]);
            }
        }
        stage ^= 1;
    }

    #pragma unroll
    for (unsigned int i = 0; i < WMMA_WM / 16; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < WMMA_WN / 16; ++j)
            wmma::store_matrix_sync(c + size_t(blockY + warpY * WMMA_WM + i * 16) * w
                                   + blockX + warpX * WMMA_WN + j * 16,
                                   acc[i][j], w, wmma::mem_row_major);
    }
}
} // namespace

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize& workSize,
                         const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                         gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k)
{
    (void)workSize;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    if (w && h) {
        const unsigned paddedH = ((h + WMMA_BM - 1) / WMMA_BM) * WMMA_BM;
        const unsigned paddedW = ((w + WMMA_BN - 1) / WMMA_BN) * WMMA_BN;
        const unsigned paddedK = k ? ((k + WMMA_BK - 1) / WMMA_BK) * WMMA_BK : WMMA_BK;
        auto& pool = context.memoryPool();
        auto aHalf = pool.allocateN<half>("matrix.wmma.a", size_t(paddedH) * paddedK);
        auto bHalf = pool.allocateN<half>("matrix.wmma.b", size_t(paddedK) * paddedW);
        const bool direct = w == paddedW && h == paddedH
                            && reinterpret_cast<std::uintptr_t>(c.cuptr()) % 32 == 0;
        auto output = direct ? c : pool.allocateN<float>("matrix.wmma.c", size_t(paddedH) * paddedW);
        wmma_convert_inputs<false><<<dim3(paddedW / 32, paddedK / 32), dim3(8, 32), 0, stream>>>(
            k ? b.cuptr() : nullptr, bHalf.cuptr(), k, w, paddedK, paddedW);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        wmma_convert_inputs<true><<<dim3(paddedK / 32, paddedH / 32), dim3(8, 32), 0, stream>>>(
            k ? a.cuptr() : nullptr, aHalf.cuptr(), h, k, paddedH, paddedK);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        matrix_multiply_wmma_pipeline<<<dim3(paddedH / WMMA_BM, paddedW / WMMA_BN),
            WMMA_THREADS, 0, stream>>>(aHalf.cuptr(), bHalf.cuptr(), output.cuptr(), paddedW, paddedH, paddedK);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        if (!direct) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(c.cuptr(), size_t(w) * sizeof(float),
                output.cuptr(), size_t(paddedW) * sizeof(float), size_t(w) * sizeof(float), h,
                cudaMemcpyDeviceToDevice, stream));
        }
    }
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
