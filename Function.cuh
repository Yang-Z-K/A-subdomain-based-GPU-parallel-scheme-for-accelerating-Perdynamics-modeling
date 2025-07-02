#ifndef PD_PARALLEL_BASE_FUNCTION_CUH
#define PD_PARALLEL_BASE_FUNCTION_CUH

#include "Base.cuh"

void initialil_parameter(device_parameter& p, BaseModel_Parameter& bp);

void set_crack_cpu(int* NN, int* NL, int* fail, real* x, real* y);

__global__ void set_crack_gpu(device_parameter p, int* NN, int* NL, int* fail, real* x, real* y);

__global__ void set_crack_ccl(device_parameter p, int* loop, unsigned int aero, int* NN, int* NL, int* fail, real* x, real* y);

#endif //PD_PARALLEL_BASE_FUNCTION_CUH