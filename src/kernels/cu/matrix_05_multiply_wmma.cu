#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>
#include <libgpu/device_memory_pool.h>
#include <libbase/stats.h>
#include <libbase/timer.h>
#include <libgpu/cuda/cu/common.cu>
#include "helpers/rassert.cu"
#include "../defines.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <tuple>

namespace {
namespace wmma = nvcuda::wmma;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
constexpr bool WMMA_VOLTA = true;
#else
constexpr bool WMMA_VOLTA = false;
#endif
constexpr unsigned int WMMA_BM = 128, WMMA_BN = 64, WMMA_BK = 32;
constexpr unsigned int WMMA_WM = 64, WMMA_WN = 32, WMMA_THREADS = 128;
constexpr unsigned int WMMA_AP = WMMA_BM + (WMMA_VOLTA ? 0 : 16), WMMA_BP = WMMA_BN + (WMMA_VOLTA ? 0 : 16);
constexpr unsigned int WMMA_AS = WMMA_BK * WMMA_AP, WMMA_BS = WMMA_BK * WMMA_BP;

// Кэш для неизменных входов; после записи/освобождения A/B или очистки pool нужен clear_cache().
using WmmaInputKey = std::tuple<const gpu::device_memory_pool*, cudaStream_t,
                               const float*, const float*, const half*, const half*, unsigned, unsigned, unsigned>;
thread_local WmmaInputKey cachedWmmaInputs{};

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
    if constexpr (WMMA_VOLTA) {
        constexpr unsigned tileWidth = TRANSPOSE ? WMMA_BM : WMMA_BN;
        const unsigned kTiles = (TRANSPOSE ? paddedCols : paddedRows) / WMMA_BK;
        const size_t tile = size_t(col / tileWidth) * kTiles + row / WMMA_BK;
        const unsigned r = col % tileWidth, t = row % WMMA_BK;
        const unsigned offset = (t / 8) * tileWidth * 8 + (r / 16) * 128
                              + ((r / 4) % 2) * 64 + ((r / 8) % 2) * 32
                              + (t % 4) * 8 + ((t / 4) % 2) * 4 + r % 4;
        half2* packed = reinterpret_cast<half2*>(dst + tile * WMMA_BK * tileWidth + offset);
        packed[0] = __floats2half2_rn(value.x, value.y);
        packed[1] = __floats2half2_rn(value.z, value.w);
    } else {
        const size_t stride = TRANSPOSE ? paddedRows : paddedCols;
        half2* packed = reinterpret_cast<half2*>(dst + size_t(row) * stride + col);
        packed[0] = __floats2half2_rn(value.x, value.y);
        packed[1] = __floats2half2_rn(value.z, value.w);
    }
}

__global__ __launch_bounds__(WMMA_THREADS, WMMA_VOLTA ? 4 : 3)
void matrix_multiply_wmma_pipeline(const half* __restrict__ a, const half* __restrict__ b, float* __restrict__ c,
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

    const half* aPanel = a + size_t(blockIdx.x) * k * WMMA_BM;
    const half* bPanel = b + size_t(blockIdx.y) * k * WMMA_BN;
    auto prefetch = [&](unsigned int baseK) {
        #pragma unroll
        for (unsigned int p = 0; p < LOADS; ++p) {
            const unsigned int index = (tid + p * WMMA_THREADS) * 8;
            if (p < LOAD_A) nextA[p] = *reinterpret_cast<const int4*>(
                WMMA_VOLTA ? aPanel + size_t(baseK) * WMMA_BM + index :
                a + size_t(baseK + index / WMMA_BM) * h + blockY + index % WMMA_BM);
            if (p < LOAD_B) nextB[p] = *reinterpret_cast<const int4*>(
                WMMA_VOLTA ? bPanel + size_t(baseK) * WMMA_BN + index :
                b + size_t(baseK + index / WMMA_BN) * w + blockX + index % WMMA_BN);
        }
    };

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    const unsigned lane = tid % 32, group = (lane / 4) % 4;
    const unsigned groupRow = (group / 2) * 8, groupCol = (group % 2) * 8;
    const unsigned halfGroup = (lane / 16) * 4;
    float acc[WMMA_WM / 16][WMMA_WN / 16][8] = {};
#else
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WMMA_WM / 16][WMMA_WN / 16];
    #pragma unroll
    for (unsigned int i = 0; i < WMMA_WM / 16; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < WMMA_WN / 16; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    }

#endif

    prefetch(0);
    unsigned int stage = 0;
    for (unsigned int baseK = 0; baseK < k; baseK += WMMA_BK) {
        half* tileA = aBase + stage * WMMA_AS;
        half* tileB = bBase + stage * WMMA_BS;
        #pragma unroll
        for (unsigned int p = 0; p < LOADS; ++p) {
            const unsigned int index = (tid + p * WMMA_THREADS) * 8;
            if (p < LOAD_A) *reinterpret_cast<int4*>(
                tileA + (WMMA_VOLTA ? index : (index / WMMA_BM) * WMMA_AP + index % WMMA_BM)) = nextA[p];
            if (p < LOAD_B) *reinterpret_cast<int4*>(
                tileB + (WMMA_VOLTA ? index : (index / WMMA_BN) * WMMA_BP + index % WMMA_BN)) = nextB[p];
        }
        __syncthreads();
        if (baseK + WMMA_BK < k) prefetch(baseK + WMMA_BK);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
        #pragma unroll
        for (unsigned q = 0; q < WMMA_BK; q += 8) {
            uint4 br[WMMA_WN / 16];
            #pragma unroll
            for (unsigned j = 0; j < WMMA_WN / 16; ++j)
                br[j] = *reinterpret_cast<const uint4*>(tileB + (q / 8) * WMMA_BN * 8
                    + (warpX * WMMA_WN / 16 + j) * 128 + (groupCol / 8) * 32
                    + (halfGroup / 4) * 64 + (lane % 4) * 8);
            #pragma unroll
            for (unsigned i = 0; i < WMMA_WM / 16; ++i) {
                const uint4 ar = *reinterpret_cast<const uint4*>(tileA + (q / 8) * WMMA_BM * 8
                    + (warpY * WMMA_WM / 16 + i) * 128 + (groupRow / 8) * 32
                    + (halfGroup / 4) * 64 + (lane % 4) * 8);
                #pragma unroll
                for (unsigned halfK = 0; halfK < 2; ++halfK) {
                    #pragma unroll
                    for (unsigned j = 0; j < WMMA_WN / 16; ++j) {
                        float* d = acc[i][j];
                        const uint2 aa = halfK ? make_uint2(ar.z, ar.w) : make_uint2(ar.x, ar.y);
                        const uint2 bb = halfK ? make_uint2(br[j].z, br[j].w) : make_uint2(br[j].x, br[j].y);
                        asm volatile("mma.sync.aligned.m8n8k4.col.row.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                            "{%0,%1,%2,%3,%4,%5,%6,%7};"
                            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
                              "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
                            : "r"(aa.x), "r"(aa.y), "r"(bb.x), "r"(bb.y));
                    }
                }
            }
        }

#else
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
    
#endif
        stage ^= 1;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700
    #pragma unroll
    for (unsigned i = 0; i < WMMA_WM / 16; ++i) {
        #pragma unroll
        for (unsigned j = 0; j < WMMA_WN / 16; ++j) {
            #pragma unroll
            for (unsigned r = 0; r < 4; r += 2) {
                const float* d = acc[i][j] + r;
                const bool right = lane & 2;
                const float a = right ? d[4] : d[0], b = right ? d[5] : d[1];
                const float u = __shfl_xor_sync(0xffffffffu, right ? d[0] : d[4], 2);
                const float v = __shfl_xor_sync(0xffffffffu, right ? d[1] : d[5], 2);
                const unsigned row = blockY + warpY * WMMA_WM + i * 16 + groupRow + halfGroup + (lane & 1) + r;
                const unsigned col = blockX + warpX * WMMA_WN + j * 16 + groupCol + (right ? 4 : 0);
                __stcs(reinterpret_cast<float4*>(c + size_t(row) * w + col), right
                    ? make_float4(u, v, a, b) : make_float4(a, b, u, v));
            }
        }
    }

#else
    #pragma unroll
    for (unsigned int i = 0; i < WMMA_WM / 16; ++i) {
        #pragma unroll
        for (unsigned int j = 0; j < WMMA_WN / 16; ++j)
            wmma::store_matrix_sync(c + size_t(blockY + warpY * WMMA_WM + i * 16) * w
                                   + blockX + warpX * WMMA_WN + j * 16,
                                   acc[i][j], w, wmma::mem_row_major);
    }

#endif
}
} // namespace

namespace cuda {
template<bool PROFILE>
static void matrix_multiply_wmma_impl(const gpu::WorkSize& workSize,
                         const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                         gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k,
                         cudaEvent_t* events = nullptr)
{
    (void)workSize;
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    const WmmaInputKey previous = cachedWmmaInputs;
    WmmaInputKey current{};
    cachedWmmaInputs = {}; // Восстанавливаем ключ только после успешного завершения вызова.
    if (w && h) {
        const unsigned paddedH = ((h + WMMA_BM - 1) / WMMA_BM) * WMMA_BM;
        const unsigned paddedW = ((w + WMMA_BN - 1) / WMMA_BN) * WMMA_BN;
        const unsigned paddedK = k ? ((k + 31) / 32) * 32 : 32;
        auto& pool = context.memoryPool();
        auto aHalf = pool.allocateN<half>("matrix.wmma.a", size_t(paddedH) * paddedK);
        auto bHalf = pool.allocateN<half>("matrix.wmma.b", size_t(paddedK) * paddedW);
        current = {&pool, stream, k ? a.cuptr() : nullptr, k ? b.cuptr() : nullptr,
                   aHalf.cuptr(), bHalf.cuptr(), w, h, k};
        const bool prepare = current != previous;
        const bool direct = w == paddedW && h == paddedH
                            && reinterpret_cast<std::uintptr_t>(c.cuptr()) % 32 == 0;
        auto output = direct ? c : pool.allocateN<float>("matrix.wmma.c", size_t(paddedH) * paddedW);
        if constexpr (PROFILE) CUDA_SAFE_CALL(cudaEventRecord(events[0], stream));
        if (prepare) {
            wmma_convert_inputs<false><<<dim3(paddedW / 32, paddedK / 32), dim3(8, 32), 0, stream>>>(
                k ? b.cuptr() : nullptr, bHalf.cuptr(), k, w, paddedK, paddedW);
            CUDA_CHECK_KERNEL_ASYNC(stream);
        }
        if constexpr (PROFILE) CUDA_SAFE_CALL(cudaEventRecord(events[1], stream));
        if (prepare) {
            wmma_convert_inputs<true><<<dim3(paddedK / 32, paddedH / 32), dim3(8, 32), 0, stream>>>(
                k ? a.cuptr() : nullptr, aHalf.cuptr(), h, k, paddedH, paddedK);
            CUDA_CHECK_KERNEL_ASYNC(stream);
        }
        if constexpr (PROFILE) CUDA_SAFE_CALL(cudaEventRecord(events[2], stream));
        matrix_multiply_wmma_pipeline<<<dim3(paddedH / WMMA_BM, paddedW / WMMA_BN),
            WMMA_THREADS, 0, stream>>>(aHalf.cuptr(), bHalf.cuptr(), output.cuptr(), paddedW, paddedH, paddedK);
        CUDA_CHECK_KERNEL_ASYNC(stream);
        if constexpr (PROFILE) CUDA_SAFE_CALL(cudaEventRecord(events[3], stream));
        if (!direct) {
            CUDA_SAFE_CALL(cudaMemcpy2DAsync(c.cuptr(), size_t(w) * sizeof(float),
                output.cuptr(), size_t(paddedW) * sizeof(float), size_t(w) * sizeof(float), h,
                cudaMemcpyDeviceToDevice, stream));
        }
        if constexpr (PROFILE) CUDA_SAFE_CALL(cudaEventRecord(events[4], stream));
    }
    CUDA_CHECK_KERNEL(stream);
    cachedWmmaInputs = current;
}

void matrix_multiply_wmma_clear_cache() { cachedWmmaInputs = {}; }

void matrix_multiply_wmma(const gpu::WorkSize& workSize,
                         const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                         gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k)
{
    matrix_multiply_wmma_impl<false>(workSize, a, b, c, w, h, k);
}

// Временный отдельный замер: вызывается после обычных тестов лабораторной.
void matrix_multiply_wmma_diagnose(const gpu::WorkSize& workSize,
                                 const gpu::gpu_mem_32f& a, const gpu::gpu_mem_32f& b,
                                 gpu::gpu_mem_32f& c, unsigned w, unsigned h, unsigned k)
{
    if (!w || !h) return;
    struct EventSet {
        cudaEvent_t items[5]{};
        ~EventSet() { for (auto event : items) if (event) cudaEventDestroy(event); }
    } events;
    for (auto& event : events.items) CUDA_SAFE_CALL(cudaEventCreate(&event));
    std::vector<double> samples[6];
    for (int repeat = -1; repeat < 10; ++repeat) {
        timer t;
        matrix_multiply_wmma_impl<true>(workSize, a, b, c, w, h, k, events.items);
        const double hostUs = t.elapsed() * 1e6;
        if (repeat < 0) continue; // Один дополнительный прогрев только диагностики.
        for (int part = 0; part < 5; ++part) {
            float ms = 0;
            CUDA_SAFE_CALL(cudaEventElapsedTime(&ms, events.items[part == 4 ? 0 : part],
                                               events.items[part == 4 ? 4 : part + 1]));
            samples[part].push_back(ms * 1000.0);
        }
        samples[5].push_back(hostUs);
    }
    const char* names[] = {"prepare B", "prepare A", "GEMM", "output copy (if needed)",
                           "GPU sequence total", "instrumented host wrapper"};
    std::printf("WMMA diagnostic: medians of 10 extra calls, microseconds; includes event/launch gaps.\n");
    for (int part = 0; part < 6; ++part)
        std::printf("  %s: %.3f us\n", names[part], stats::median(samples[part]));
}
} // namespace cuda
