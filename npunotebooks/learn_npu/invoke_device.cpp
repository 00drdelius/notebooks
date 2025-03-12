#include "general.h"

namespace npu_learning
{

aclError return_npuDevice(int32_t *idx){
    aclError error = c10_npu::GetDevice(idx);
    std::cout << "error from c10_npu::GetDevice"
    << error << std::endl;
    std::cout << '*'*50 << std::endl;
    // NPU_CHECK_ERROR(error);
    return error;
}

}