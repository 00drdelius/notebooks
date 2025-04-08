// https://github.com/xlite-dev/CUDA-Learn-Notes/blob/main/kernels/elementwise/elementwise.cu
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
// #include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <torch/types.h>
// #include <torch/extension.h>

#include <cuda_learning_utils.h>

#define WARP_SIZE 32

/**
 * reinterpret_cast: 强制转换编译时对变量位模式的解释类型（对其底层二进制代码的解释）。如：
 * ```cpp
 * int num=6513249; // 6513249(10) = 0x00636261(16)
 * int *pnum=&num; 
 * // pnum是int指针，int为4字节。位模式下 *pnum 会直接读取指针指向地址的连续4字节的数据，再翻译成int，即6513249。
 * 
 * char *pstr = reinterpret_cast<char*>pnum;
 * // pstr是 char指针，char为1字节。位模式下 *pstr 会直接读取指针指向地址的连续1字节的数据。
 * 因为是小端存储，所以从右往左读十六进制的两位数8个比特，即0x61，再根据ASCII码翻译得 a。
 * ```
 * 详见`cppnotebooks/reinterpret_cast.cc`
 */
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])


// -------------------------------------- FP32 -------------------------------------- 
// ElementWise Add  
// a: Nx1, b: Nx1, c: Nx1, c = elementwise_add(a, b)
__global__ void elementwise_add_fp32_kernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N) c[idx]=a[idx]+b[idx];
}

/**
 * 2-dim torch::Tensor saves elements in mem is Row-Major Order(行优先),
 * so it can be elementwise-added in cuda in one-dim.
 */
torch::Tensor elementwise_add_fp32(const torch::Tensor &a,const torch::Tensor &b)
{
    CHECK_TORCH_TENSOR_DEVICE(a);
    CHECK_TORCH_TENSOR_DEVICE(b);
    CHECK_TORCH_TENSORS_SIZES(a,b);
    
    const int ndim = a.dim();
    const int64_t N = a.numel();
    torch::Tensor output = torch::empty_like(a);

    dim3 block_size(16);
    dim3 grid_size((N+block_size.x-1)/block_size.x);
    elementwise_add_fp32_kernel<<<grid_size, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        N
    );
    CUDA_ERROR_LOG

    return output;
}

// --------------------- PyTorch bindings for custom kernel -----------------------

TORCH_PYBIND(elementwise_add_fp32)

// TORCH_LIBRARY(custom_ops, m){m.def("add_fp32",torch_add_fp32);}