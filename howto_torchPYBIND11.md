# how to PYBIND11 in torch
Let's simply start with an example:
```cpp
#include <cuda_runtime.h>

#include <torch/types.h>
#include <torch/extension.h>

__global__ void elementwise_add_fp32_kernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx<N) c[idx]=a[idx]+b[idx];
}

void elementwise_add_fp32(const float *a, const float *b,float *c, int N)
{
    dim3 block_size(16), grid_size(N/16);
    elementwise_add_fp32_kernel<<<grid_size, block_size>>>(a, b, c, N);
}

void torch_add_fp32(const torch::Tensor &a,const torch::Tensor &b, torch::Tensor &c, int64_t N)
{
    elementwise_add_fp32(
        (const float*)a.data_ptr(),
        (const float*)b.data_ptr(),
        (float*)c.data_ptr(),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "torch_add_fp32",
        &torch_add_fp32,
        "add two one-dim tensors into one"
    );
}

TORCH_LIBRARY(custom_ops, m){m.def("add_fp32",torch_add_fp32);}
```

The above `PYBIND11_MODULE` obviously is the key. But it's a complicated marco function. However, we can analyse the preprocess stage to find out what does it do:
```shell
# specify include search path
export CPLUS_INCLUDE_PATH=\
$PATH_TO_PYTORCH/include/torch/csrc/api/include:\
$PATH_TO_PYTORCH/include:\
$PATH_TO_PYTHON_INCLUDE:\
$CPLUS_INCLUDE_PATH

# preprocess .cu file, output verbosely(for checking include path and others)
nvcc above_code.cu -E -o above_code.i --verbose -Xcompiler -v
```

then we can get preprocessed .i file, in which all macro functions is expanded:
```cpp
//omit expanded 360k codes

# 57 "elementwise.cu"
void torch_add_fp32(const torch::Tensor &a,const torch::Tensor &b, torch::Tensor &c, int64_t N)
{
    elementwise_add_fp32(
        (const float*)a.data_ptr(),
        (const float*)b.data_ptr(),
        (float*)c.data_ptr(),
        N
    );
}

# 67 "elementwise.cu" 3
static ::pybind11::module_::module_def pybind11_module_def_TORCH_EXTENSION_NAME [[maybe_unused]]; [[maybe_unused]] static void pybind11_init_TORCH_EXTENSION_NAME(::pybind11::module_ &); extern "C" [[maybe_unused]] __attribute__((visibility("default"))) PyObject *PyInit_TORCH_EXTENSION_NAME(); extern "C" __attribute__((visibility("default"))) PyObject *PyInit_TORCH_EXTENSION_NAME() { { const char *compiled_ver = 
# 67 "elementwise.cu"
"3" 
# 67 "elementwise.cu" 3
"." 
# 67 "elementwise.cu"
"11"
# 67 "elementwise.cu" 3
; const char *runtime_ver = Py_GetVersion(); size_t len = std::strlen(compiled_ver); if (std::strncmp(runtime_ver, compiled_ver, len) != 0 || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) { PyErr_Format(PyExc_ImportError, "Python version mismatch: module was compiled for Python %s, " "but the interpreter version is incompatible: %s.", compiled_ver, runtime_ver); return nullptr; } } pybind11::detail::get_internals(); auto m = ::pybind11::module_::create_extension_module( 
# 67 "elementwise.cu"
"TORCH_EXTENSION_NAME"
# 67 "elementwise.cu" 3
, nullptr, &pybind11_module_def_TORCH_EXTENSION_NAME); try { pybind11_init_TORCH_EXTENSION_NAME(m); return m.ptr(); } catch (pybind11::error_already_set & e) { pybind11::raise_from(e, PyExc_ImportError, "initialization failed"); return nullptr; } catch (const std::exception &e) { ::pybind11::set_error(PyExc_ImportError, e.what()); return nullptr; } } void pybind11_init_TORCH_EXTENSION_NAME(::pybind11::module_ & (
# 67 "elementwise.cu"
m
# 67 "elementwise.cu" 3
))

# 68 "elementwise.cu"
{
    m.def(
        "torch_add_fp32",
        &torch_add_fp32,
        "add two one-dim tensors into one"
    );
}


# 76 "elementwise.cu" 3
static void TORCH_LIBRARY_init_custom_ops(torch::Library&); static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_custom_ops( torch::Library::DEF, &TORCH_LIBRARY_init_custom_ops, 
# 76 "elementwise.cu"
"custom_ops"
# 76 "elementwise.cu" 3
, c10::nullopt, "elementwise.cu", 76); void TORCH_LIBRARY_init_custom_ops(torch::Library& 
# 76 "elementwise.cu"
m
# 76 "elementwise.cu" 3
)
# 76 "elementwise.cu"
                           {m.def("add_fp32",torch_add_fp32);}

```

remove all lines starts with # and format codes:
```cpp
//omit explicit function definition
void elementwise_add_fp32(...);

void torch_add_fp32(const torch::Tensor &a,const torch::Tensor &b, torch::Tensor &c, int64_t N)
{
    elementwise_add_fp32(
        (const float*)a.data_ptr(),
        (const float*)b.data_ptr(),
        (float*)c.data_ptr(),
        N
    );
}

// ------------------------ PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) ------------------------
/**
 * 定义了静态的 module_def 结构体，用于存储模块的元信息
 * [[maybe_unused]]：一种属性（稍后提及）：
 * 通常情况下，对于声明了但是从未使用过的变量会给出警告信息。
 * 但是在声明的时候添加了这个属性，则编译器确认是程序故意为之的逻辑，则不再发出警告。
 * 但需要注意，这个声明不会影响编译器的优化逻辑，在编译优化阶段，无用的变量该干掉还是会被干掉的。
 *  */
static ::pybind11::module_::module_def pybind11_module_def_TORCH_EXTENSION_NAME [[maybe_unused]];

// 声明了模块初始化函数 pybind11_init_TORCH_EXTENSION_NAME
[[maybe_unused]] static void pybind11_init_TORCH_EXTENSION_NAME(::pybind11::module_ &);

/**
 * 声明了 Python 模块入口函数 PyInit_TORCH_EXTENSION_NAME
 * __attribute__((visibility("default")))：同[[maybe_unused]]，也是一种属性（稍后提及），但是仅GNU支持。
 *  */
extern "C" [[maybe_unused]] __attribute__((visibility("default"))) PyObject *PyInit_TORCH_EXTENSION_NAME();
extern "C" __attribute__((visibility("default"))) PyObject *PyInit_TORCH_EXTENSION_NAME() {
    // Python 版本检查
    { 
        const char *compiled_ver = "3.11";
        const char *runtime_ver = Py_GetVersion();
        size_t len = std::strlen(compiled_ver);
        if (std::strncmp(runtime_ver, compiled_ver, len) != 0 ||
        (runtime_ver[len] >= '0' && runtime_ver[len] <= '9'))
        {
            PyErr_Format(
                PyExc_ImportError,
                "Python version mismatch: module was compiled for Python %s, "
                "but the interpreter version is incompatible: %s.",
                compiled_ver, runtime_ver
            ); return nullptr;
        }
    }
    // 初始化 pybind11 内部状态
    pybind11::detail::get_internals();
    // 创建扩展模块
    auto m = ::pybind11::module_::create_extension_module(
        "TORCH_EXTENSION_NAME",
        nullptr,
        &pybind11_module_def_TORCH_EXTENSION_NAME
    );
    // 尝试初始化模块
    try {
        pybind11_init_TORCH_EXTENSION_NAME(m);
        return m.ptr();
    } catch (pybind11::error_already_set & e) {
        pybind11::raise_from(e, PyExc_ImportError, "initialization failed");
        return nullptr;
    } catch (const std::exception &e) {
        ::pybind11::set_error(PyExc_ImportError, e.what());
        return nullptr;
    }
}

void pybind11_init_TORCH_EXTENSION_NAME(::pybind11::module_ & (m))
{
    m.def(
        "torch_add_fp32",
        &torch_add_fp32,
        "add two one-dim tensors into one"
    );
}
// ------------------------ PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) ------------------------

// ------------------------ TORCH_LIBRARY(custom_ops, m) ------------------------
static void TORCH_LIBRARY_init_custom_ops(torch::Library&);

static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_custom_ops(
    torch::Library::DEF,
    &TORCH_LIBRARY_init_custom_ops, 
    "custom_ops",
    c10::nullopt,
    "elementwise.cu",
    76);

void TORCH_LIBRARY_init_custom_ops(torch::Library& m){
    m.def("add_fp32",torch_add_fp32);
}
// ------------------------ TORCH_LIBRARY(custom_ops, m) ------------------------

```

## cpp 属性
> refer to [谈谈C++新标准带来的属性（Attribute）](https://zhuanlan.zhihu.com/p/392460397)
c++03开始支持的一种语法：一种特殊的用户命名空间——“双下划线关键词”。较为广泛使用的有：
- GNU和IBM的`__attribute__ `
- 微软的`__declspec()`

上述代码的`__attribute__(visibility("default"))`就是旧时代的产物。用于设置动态链接库中函数的可见性，以有效避免so之间的符号冲突。
- `__attribute__(visibility("default))`则该符号外部可见
- `__attribute__(visibility("hidden"))`会将变量或函数设置为hidden，则该符号仅在本so中可见，在其他库中则不可见
- 若编译器设置`-fvisibility=hidden`则全部符号默认设置为hidden；否则全部默认default，对外部可见
- 用`nm`或`readelf`命令输出该文件的动态符号表时，会有特殊标记标志 hidden 的符号为不可见。此时动态链接时，若有该符号的调用，会引发 undefined reference 错误。

随着各大厂商编译器及语言标准的发展，各个厂商的编译器在“属性”这个标准上渐行渐远，互不兼容。
因此，c++11开始，使用“双中括号”的属性新标准应运而生。  
根据该标准，双中括号扩起的属性可以：
- 修饰：函数，变量，函数或者变量的名称，类型，程序块，编译单元
- 可以出现在程序内几乎任何位置：`[[attr1]] class C [[ attr2 ]] { } [[ attr3 ]] c [[ attr4 ]], d [[ attr5 ]];`
    - attr1 作用于class C的实体定义c和d
    - attr2 作用于class C的定义
    - attr3 作用于类型C
    - attr4 作用于实体c
    - attr5 作用于实体d

而上述代码中使用的`[[maybe_unused]]`属性在c++17的标准中提出：  
`[[maybe_unused]]`：通常情况下，对于声明了但是从未使用过的变量会给出警告信息。
但是在声明的时候添加了这个属性，则编译器确认是程序故意为之的逻辑，则不再发出警告。
需要注意的是，这个声明不会影响编译器的优化逻辑，在编译优化阶段，无用的变量该干掉还是会被干掉的。