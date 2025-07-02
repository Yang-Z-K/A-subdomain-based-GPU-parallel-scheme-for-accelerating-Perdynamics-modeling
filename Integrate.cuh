#ifndef PD_PARALLEL_INTEGRATE_CUH
#define PD_PARALLEL_INTEGRATE_CUH
#include "Base.cuh"

//CPU
void boundary_cpu(int tt, real* y, real* disp_y, real* vel_y);

void updata_vel_cpu(
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y);

//GPU
__global__ void boundary_gpu(device_parameter p, int tt, real* y, real* disp_y, real* vel_y);

__global__ void updata_vel_gpu(device_parameter p,
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y);

//CCL
__global__ void boundary_ccl(device_parameter p, int* loop, unsigned int aero, int tt, real* y, real* disp_y, real* vel_y);

__global__ void updata_vel_ccl(device_parameter p, int* loop, unsigned int aero,
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y);

#endif //PD_PARALLEL_INTEGRATE_CUH