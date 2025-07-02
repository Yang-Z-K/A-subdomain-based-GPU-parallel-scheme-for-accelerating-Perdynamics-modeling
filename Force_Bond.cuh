#ifndef PD_PARALLEL_FORCE_BOND_CUH
#define PD_PARALLEL_FORCE_BOND_CUH

#include "Base.cuh"
//CPU
void bond_force_cpu(
    real* x, real* disp_x, real* fncst_x, real* pforce_x,
    real* y, real* disp_y, real* fncst_y, real* pforce_y,
    int* NN, int* NL, int* fail, real* dmg);

//GPU
__global__ void bond_force_gpu(device_parameter p,
    real* x, real* disp_x, real* fncst_x, real* pforce_x,
    real* y, real* disp_y, real* fncst_y, real* pforce_y,
    int* NN, int* NL, int* fail, real* dmg);

//CCL
__global__ void bond_force_ccl(device_parameter p, int* loop, unsigned int aero,
    real* x, real* disp_x, real* fncst_x, real* pforce_x,
    real* y, real* disp_y, real* fncst_y, real* pforce_y,
    int* NN, int* NL, int* fail, real* dmg);

#endif //PD_PARALLEL_FORCE_BOND_CUH