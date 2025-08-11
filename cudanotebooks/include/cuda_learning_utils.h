#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

#define WARP_SIZE 32

#define STRINGFY(str) #str

#define CHECK_TORCH_TENSOR_DEVICE(t)\
AT_ASSERTM(t.is_cuda(), #t" must be a CUDA tensor");

#define CHECK_TORCH_TENSORS_SIZES(t1,t2)\
AT_ASSERTM(t1.sizes()==t2.sizes(),"size of "#t1" and "#t2" must be equal");

#define CUDA_ERROR_LOG \
{\
    cudaError_t error_code = cudaGetLastError();\
    if (error_code!=cudaSuccess){\
        std::cout\
        << "File: " << __FILE__ << "\n"\
        << "Line: " << __LINE__ << "\n"\
        << "Error Code: " << error_code << "\n"\
        << "Error String: " << cudaGetErrorString(error_code) << std::endl;\
    }\
};

#define TORCH_PYBIND(func)\
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)\
{\
    m.def(\
        #func,\
        &func,\
        #func\
    );\
}
