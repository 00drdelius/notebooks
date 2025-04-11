#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#define SPECS(prop_spec)\
std::cout<<#prop_spec": "<<prop_spec<<std::endl;

#define SPECS_ARRAY(prop_spec)\
std::cout<< #prop_spec": " <<"(";\
for(int i:prop_spec){std::cout << i << ", ";}\
std::cout<<")"<<std::endl;\

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    SPECS(prop.name)
    SPECS(prop.totalGlobalMem)
    SPECS(prop.sharedMemPerBlock) //49152
    SPECS(prop.regsPerBlock) //65536
    SPECS(prop.warpSize) //32
    SPECS(prop.maxThreadsPerBlock) //1024
    SPECS(prop.maxBlocksPerMultiProcessor) //16
    SPECS_ARRAY(prop.maxGridSize)
    SPECS_ARRAY(prop.maxThreadsDim)
    std::cin.get();
}