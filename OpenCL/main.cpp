#include <iostream>
#include <chrono>
#include "CL/cl.hpp"

#define PRINT_INFO(obj, info) \
    std::cout << #info << " : " << (obj).getInfo<info>() << '\n';

#define PRINT_VECTOR(obj, info) \
    std::cout << #info << " : "; \
    for (const auto& el : (obj).getInfo<info>()) std::cout << el << ' '; \
    std::cout << '\n';

#define CHECK(err) \
    if (err) { std::cerr << "Error(" << err << ") file " << __FILE__ << " line " << __LINE__ << '\n'; exit(err); }

#define LOG_BUILD(prog, dev, err) \
    if (err) std::cerr << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev) << '\n';


std::chrono::time_point<std::chrono::high_resolution_clock> time_point = std::chrono::high_resolution_clock::now();
#define TIME_POINT { \
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now()-time_point).count(); \
    std::cout << "time " << time << " line "<< __LINE__ << '\n'; \
    time_point = std::chrono::high_resolution_clock::now(); \
}

void print_info()
{
    std::vector<cl::Platform> cl_platforms;
    cl::Platform::get(&cl_platforms);
    for (const auto& platform : cl_platforms)
    {
        PRINT_INFO(platform, CL_PLATFORM_PROFILE);
        PRINT_INFO(platform, CL_PLATFORM_VERSION);
        PRINT_INFO(platform, CL_PLATFORM_NAME);
        PRINT_INFO(platform, CL_PLATFORM_VENDOR);
        PRINT_INFO(platform, CL_PLATFORM_EXTENSIONS);
        std::cout << '\n';

        std::vector<cl::Device> cl_devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &cl_devices);
        for (const auto& device : cl_devices)
        {
            PRINT_INFO(device, CL_DEVICE_TYPE);
            PRINT_INFO(device, CL_DEVICE_VENDOR_ID);
            PRINT_INFO(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            PRINT_INFO(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
            PRINT_INFO(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            PRINT_VECTOR(device, CL_DEVICE_MAX_WORK_ITEM_SIZES);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            PRINT_INFO(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            PRINT_INFO(device, CL_DEVICE_ADDRESS_BITS);
            PRINT_INFO(device, CL_DEVICE_MAX_READ_IMAGE_ARGS);
            PRINT_INFO(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
            PRINT_INFO(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            PRINT_INFO(device, CL_DEVICE_IMAGE2D_MAX_WIDTH);
            PRINT_INFO(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
            PRINT_INFO(device, CL_DEVICE_IMAGE3D_MAX_WIDTH);
            PRINT_INFO(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
            PRINT_INFO(device, CL_DEVICE_IMAGE3D_MAX_DEPTH);
            PRINT_INFO(device, CL_DEVICE_IMAGE_SUPPORT);
            PRINT_INFO(device, CL_DEVICE_MAX_PARAMETER_SIZE);
            PRINT_INFO(device, CL_DEVICE_MAX_SAMPLERS);
            PRINT_INFO(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN);
            PRINT_INFO(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
            PRINT_INFO(device, CL_DEVICE_SINGLE_FP_CONFIG);
            PRINT_INFO(device, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE);
            PRINT_INFO(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
            PRINT_INFO(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
            PRINT_INFO(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            PRINT_INFO(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            PRINT_INFO(device, CL_DEVICE_MAX_CONSTANT_ARGS);
            PRINT_INFO(device, CL_DEVICE_LOCAL_MEM_TYPE);
            PRINT_INFO(device, CL_DEVICE_LOCAL_MEM_SIZE);
            PRINT_INFO(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            PRINT_INFO(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION);
            PRINT_INFO(device, CL_DEVICE_ENDIAN_LITTLE);
            PRINT_INFO(device, CL_DEVICE_AVAILABLE);
            PRINT_INFO(device, CL_DEVICE_COMPILER_AVAILABLE);
            PRINT_INFO(device, CL_DEVICE_EXECUTION_CAPABILITIES);
            PRINT_INFO(device, CL_DEVICE_QUEUE_ON_HOST_PROPERTIES);
            PRINT_INFO(device, CL_DEVICE_NAME);
            PRINT_INFO(device, CL_DEVICE_VENDOR);
            PRINT_INFO(device, CL_DRIVER_VERSION);
            PRINT_INFO(device, CL_DEVICE_PROFILE);
            PRINT_INFO(device, CL_DEVICE_VERSION);
            PRINT_INFO(device, CL_DEVICE_EXTENSIONS);
            PRINT_INFO(device, CL_DEVICE_PLATFORM);
            PRINT_INFO(device, CL_DEVICE_DOUBLE_FP_CONFIG);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
            PRINT_INFO(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF);
            PRINT_INFO(device, CL_DEVICE_OPENCL_C_VERSION);
            PRINT_INFO(device, CL_DEVICE_BUILT_IN_KERNELS);
            PRINT_INFO(device, CL_DEVICE_PARENT_DEVICE);
            PRINT_INFO(device, CL_DEVICE_PARTITION_AFFINITY_DOMAIN);
            PRINT_INFO(device, CL_DEVICE_REFERENCE_COUNT);
            PRINT_INFO(device, CL_DEVICE_PREFERRED_INTEROP_USER_SYNC);
            std::cout << '\n';
        }
    }
}

#define at(mat, i, j) (mat)[(i) * (size) + (j)]

void mat_mul(float* a, float* res, size_t size)
{
    for (size_t i = 0; i < size; i++)
    {
        for (size_t j = 0; j < size; j++)
        {
            at(res, i, j) = 0;
            for (size_t k = 0; k < size; k++)
            {
                at(res, i, j) += at(a, i, k) * at(a, k, j);
            }
        }
    }
}

std::string str_kernel = R"(
#define at(mat, i, j) (mat)[(i) * (size) + (j)]
kernel void main_kernel(
    global float* data,
    global float* out
)
{
    size_t size = get_global_size(0);
    size_t i = get_global_id(0);
    for (size_t j = 0; j < size; j++)
    {
        at(out, i, j) = 0;
        for (size_t k = 0; k < size; k++)
        {
            at(out, i, j) += at(data, i, k) * at(data, k, j);
        }
    }
}
)";

constexpr size_t N = 1 << 11;

int main()
{
    cl_int error;
    print_info();

    /*

    TIME_POINT;

    std::vector<cl::Platform> cl_platforms;
    cl::Platform::get(&cl_platforms);
    std::vector<cl::Device> cl_devices;
    cl_platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &cl_devices);
    cl::Device device = cl_devices[0];

    TIME_POINT;

    cl::Context context(device, NULL, NULL, NULL, &error); CHECK(error);
    cl::CommandQueue queue(context, device, NULL, &error); CHECK(error);
    cl::Program program(context, str_kernel.c_str(), false, &error); CHECK(error);
    error = program.build("-cl-std=CL3.0"); LOG_BUILD(program, device, error); CHECK(error);
    cl::Kernel kernel(program, "main_kernel", &error); CHECK(error);

    TIME_POINT;

    std::vector<float> in_vec(N * N, 1.0F);
    std::vector<float> out_vec(N * N, 0.0F);

    TIME_POINT;

    cl::Buffer in_buf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, in_vec.data(), &error); CHECK(error);
    cl::Buffer out_buf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(float) * N * N, NULL, &error); CHECK(error);

    TIME_POINT;

    error = kernel.setArg(0, in_buf); CHECK(error);
    error = kernel.setArg(1, out_buf); CHECK(error);

    TIME_POINT;

    error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N), cl::NDRange(device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>())); CHECK(error);
    error = queue.finish(); CHECK(error);

    TIME_POINT;

    error = queue.enqueueReadBuffer(out_buf, true, 0, sizeof(float) * N * N, out_vec.data());

    TIME_POINT;

    mat_mul(in_vec.data(), out_vec.data(), N);

    TIME_POINT;
    */

    return 0;
}
