#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <exception>
#ifdef CUDA_SUPPORT
#include <libgpu/cuda/cuda_api.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<unsigned> scanOnDevices(const std::vector<gpu::Device>& devices,
    const std::vector<unsigned>& input, std::vector<double>& times, int iterations = 10)
{
    rassert(!devices.empty() && iterations > 0
        && input.size() <= std::numeric_limits<unsigned>::max(), 7419301043);
    const int requested = int(std::min({devices.size(), std::max(size_t(1), input.size()), size_t(4)}));
    std::vector<unsigned> output(input.size());
    std::vector<const unsigned*> totals(requested);
    std::vector<cudaContext_t> peer_contexts(requested);
    std::vector<std::exception_ptr> failures(requested);
    bool ready = false;
    times.resize(iterations);
    timer t(true);
    #pragma omp parallel num_threads(requested) if(requested > 1)
    {
        int g = 0, count = 1;
#ifdef _OPENMP
        g = omp_get_thread_num();
        count = omp_get_num_threads();
#endif
        const size_t begin = input.size() * g / count;
        const unsigned n = input.size() * (g + 1) / count - begin;
        gpu::Context context;
        gpu::gpu_mem_32u in, out, total;
        auto attempt = [&](auto action) {
            if (failures[g]) return;
            try { action(); }
            catch (...) { failures[g] = std::current_exception(); }
        };
        attempt([&] {
            if (g) { context.init(devices[g].device_id_cuda); context.activate(); }
            in.resizeN(n); out.resizeN(n);
            if (n) in.writeN(input.data() + begin, n);
            if (count > 1) {
                total.resizeN(1);
                totals[g] = total.cuptr();
                CU_SAFE_CALL(cuCtxGetCurrent(&peer_contexts[g]));

            }
        });
        #pragma omp barrier
        attempt([&] {
            for (int previous = 0; previous < g; ++previous) {
                rassert(peer_contexts[previous], 7419301046);
                cuda::reportErrorCU(cuCtxEnablePeerAccess(peer_contexts[previous], 0), __LINE__,
                    "Direct GPU peer access is required (no CPU fallback): ");
            }
        });
        #pragma omp barrier
        #pragma omp single
        {
            ready = std::none_of(failures.begin(), failures.end(), [](const auto& e) { return bool(e); });
            if (ready) std::cout << "Prefix sum on " << count << " CUDA GPU(s)"
                                 << (count > 1 ? ", direct peer exchange" : "") << std::endl;
        }
        for (int iter = -1; ready && iter < iterations; ++iter) {
            #pragma omp single
            { t.restart(); }
            attempt([&] {
                cuda::prefix_sum(in, out, n, 0, count > 1 ? total.cuptr() : nullptr);
            });
            #pragma omp barrier
            attempt([&] {
                cuda::prefix_sum_add_offsets(out, n, totals.data(), g);
            });
            #pragma omp barrier
            #pragma omp single
            { if (iter >= 0) times[iter] = t.elapsed(); }
        }
        if (ready) attempt([&] {
            if (n) out.readN(output.data() + begin, n);
            const auto unchanged = in.readVector();
            for (unsigned i = 0; i < n; ++i)
                rassert(unchanged[i] == input[begin + i], 6573452432, i);
        });
#ifdef CUDA_SUPPORT
        if (failures[g]) (void)cudaGetLastError();
#endif
    }
    for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
    return output;
}

void run(int argc, char** argv)
{
    // chooseGPUVkDevices:
    // - Если не доступо ни одного устройства - кинет ошибку
    // - Если доступно ровно одно устройство - вернет это устройство
    // - Если доступно N>1 устройства:
    //   - Если аргументов запуска нет или переданное число не находится в диапазоне от 0 до N-1 - кинет ошибку
    //   - Если аргумент запуска есть и он от 0 до N-1 - вернет устройство под указанным номером
    const auto available = gpu::selectAllDevices(ALL_GPUS, true);
    gpu::Device device = gpu::chooseGPUDevice(available, argc, argv);

    // TODO 000 сделайте здесь свой выбор API - если он отличается от OpenCL то в этой строке нужно заменить TypeOpenCL на TypeCUDA или TypeVulkan
    // TODO 000 после этого изучите этот код, запустите его, изучите соответсвующий вашему выбору кернел - src/kernels/<ваш выбор>/aplusb.<ваш выбор>
    // TODO 000 P.S. если вы выбрали CUDA - не забудьте установить CUDA SDK и добавить -DGPU_CUDA_SUPPORT=ON в CMake options
    // TODO 010 P.S. так же в случае CUDA - добавьте в CMake options (НЕ меняйте сами CMakeLists.txt чтобы не менять окружение тестирования):
    // TODO 010 "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS=-lineinfo" (первое - чтобы включить поддержку WMMA, второе - чтобы compute-sanitizer и профилировщик знали номера строк кернела)
    gpu::Context context = activateContext(device, gpu::Context::TypeCUDA);
    std::vector<gpu::Device> devices{device};
    for (const auto& candidate : available)
        if (candidate.supports_cuda && candidate.device_id_cuda != device.device_id_cuda && devices.size() < 4)
            devices.push_back(candidate);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_fill_with_zeros(ocl::getFillBufferWithZeros());
    ocl::KernelSource ocl_sum_reduction(ocl::getPrefixSum01Reduction());
    ocl::KernelSource ocl_prefix_accumulation(ocl::getPrefixSum02PrefixAccumulation());

    avk2::KernelSource vk_fill_with_zeros(avk2::getFillBufferWithZeros());
    avk2::KernelSource vk_sum_reduction(avk2::getPrefixSum01Reduction());
    avk2::KernelSource vk_prefix_accumulation(avk2::getPrefixSum02PrefixAccumulation());

    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    size_t total_sum = 0;
    for (size_t i = 0; i < n; ++i) {
        as[i] = (3 * (i + 5) + 7) % 17;
        total_sum += as[i];
        rassert(total_sum < std::numeric_limits<unsigned int>::max(), 5462345234231, total_sum, as[i], i); // ensure no overflow
    }

    std::vector<unsigned int> cpu_prefix_sum(n);
    {
        std::cout << "prefix sum on CPU..." << std::endl;
        timer t;
        unsigned int sum = 0;
        for (size_t i = 0; i < n; ++i) {
            sum += as[i];
            cpu_prefix_sum[i] = sum;
        }
        double elapsed = t.elapsed();
        double memory_size_gb = sizeof(unsigned int) * 2.0 * n / 1024.0 / 1024.0 / 1024.0;
        std::cout << "CPU prefix sum finished in " << elapsed << " sec" << std::endl;
        std::cout << "CPU prefix sum effective RAM bandwidth: "
                  << memory_size_gb / elapsed << " GB/s ("
                  << n / 1e6 / elapsed << " uint millions/s)" << std::endl;
    }

    std::vector<double> times;
    const auto gpu_prefix_sum = scanOnDevices(devices, as, times);
    std::cout << "prefix sum times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

    // Вычисляем достигнутую эффективную пропускную способность видеопамяти (из соображений что мы отработали в один проход - считали массив и сохранили префиксные суммы)
    double memory_size_gb = sizeof(unsigned int) * 2 * n / 1024.0 / 1024.0 / 1024.0;
    std::cout << "GPU prefix sum median effective VRAM bandwidth: "
              << memory_size_gb / stats::median(times) << " GB/s ("
              << n / 1e6 / stats::median(times) << " uint millions/s)" << std::endl;

    // Сверяем результат
    for (size_t i = 0; i < n; ++i) {
        rassert(cpu_prefix_sum[i] == gpu_prefix_sum[i], 566324523452323, cpu_prefix_sum[i], gpu_prefix_sum[i], i);
    }

}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
