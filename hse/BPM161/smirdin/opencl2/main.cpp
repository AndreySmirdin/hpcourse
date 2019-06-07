#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__

#include <OpenCL/cl_platform.h>

#else
#include <CL/cl_platform.h>
#endif

#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <assert.h>

size_t const BLOCK_SIZE = 256;

int rounded_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

std::vector<float> fix_sums(std::vector<float>& input, std::vector<float>& block_sums,
                            cl::CommandQueue& queue, cl::Context& context, cl::Program& program) {
    int N = input.size();
    std::vector<float> output(N);
    int output_size = N;
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer dev_input_sums(context, CL_MEM_READ_ONLY, sizeof(float) * block_sums.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_size);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * N, &input[0]);
    queue.enqueueWriteBuffer(dev_input_sums, CL_TRUE, 0, sizeof(float) * block_sums.size(),
                             &block_sums[0]);
    queue.finish();

    cl::Kernel kernel_hs(program, "fix_sum");

    cl::KernelFunctor fix_sums_kernel(kernel_hs, queue, cl::NullRange, cl::NDRange(rounded_size(N)),
                                      cl::NDRange(BLOCK_SIZE));
    fix_sums_kernel(input.size(), dev_input, dev_input_sums, dev_output);

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * output_size, &output[0]);
    return output;
}

std::vector<float> calc_prefix_sums(std::vector<float>& input,
                                    cl::CommandQueue& queue, cl::Context& context,
                                    cl::Program& program) {
    int N = input.size();
    std::vector<float> output(N);
    int output_size = N;
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(float) * N);
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_size);

    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(float) * N, &input[0]);
    queue.finish();

    cl::Kernel kernel_hs(program, "scan_hillis_steele");
    cl::KernelFunctor scan_hs(kernel_hs, queue, cl::NullRange, cl::NDRange(rounded_size(N)),
                              cl::NDRange(BLOCK_SIZE));
    scan_hs(N, dev_input, dev_output, cl::__local(sizeof(float) * BLOCK_SIZE),
            cl::__local(sizeof(float) * BLOCK_SIZE));

    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(float) * output_size, &output[0]);

    if (N < BLOCK_SIZE) return output;

    int number_of_blocks = rounded_size(N) / BLOCK_SIZE;
    std::vector<float> block_sums(number_of_blocks, 0);
    for (int i = 1; i < number_of_blocks; i++) {
        block_sums[i] = output[BLOCK_SIZE * i - 1];
    }

    auto prefix_sums_blocks = calc_prefix_sums(block_sums, queue, context, program);
    return fix_sums(output, prefix_sums_blocks, queue, context, program);
}


int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file),
                              (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        cl::Program program(context, source);
        program.build(devices);


        std::ifstream ifs("input.txt");
        std::ofstream ofs("output.txt");

        int N;
        ifs >> N;

        size_t const output_size = N;
        std::vector<float> input(N);
        for (size_t i = 0; i < N; ++i) {
            ifs >> input[i];
        }

        auto output = calc_prefix_sums(input, queue, context, program);

        for (int i = 0; i < output_size; i++) {
            ofs << std::setprecision(6) << output[i] << " ";
        }
    }
    catch (cl::Error& e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
