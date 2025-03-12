#pragma once
#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include <iostream>

namespace npu_learning
{

typedef std::int32_t int32_t;

aclError return_npuDevice(int32_t *idx);

#define PARAMS_LOG(param)  \
do {  \
    std::cout  \
    << "Paramters:  "  \
    << "\n  name: " << #param  \
    << "\n  type: " << typeid(param).name()  \
    << "\n  value: " << param  \
    << std::endl;  \
} while(false)


}