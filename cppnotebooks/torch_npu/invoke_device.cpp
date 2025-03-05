#include "third_party/acl/inc/acl/acl_base.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include <iostream>

int main(){
    int deviceIdx = 0;
    aclError error(c10_npu::GetDevice(&deviceIdx));
    std::cout << "aclError: " << error << std::endl;
    std::cin.get();
    return 0;
}