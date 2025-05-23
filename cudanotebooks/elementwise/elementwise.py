import os
os.environ['TORCH_CUDA_ARCH_LIST']="Ampere"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']="1"
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

cuda_lib = load(
    "elementwise_lib",
    sources=['./elementwise.cu'],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        # "ptxas-options=-v"
    ],
    extra_cflags=["-std=c++17"],
    verbose=True,
    extra_include_paths=[
        "/home/gmcc/workspace/notebooks/cudanotebooks/include"
    ]
)
print(cuda_lib)

shape=(3,5)

def main():
    a=torch.randint(0,10,shape).cuda().float().contiguous()
    b=torch.randint(0,10,shape).cuda().float().contiguous()
    print("[a] ",a)
    print("[b] ",b)
    output:torch.Tensor=cuda_lib.elementwise_add_fp32(a,b)
    print("[output] ",output.cpu())

if __name__=='__main__':
    main()