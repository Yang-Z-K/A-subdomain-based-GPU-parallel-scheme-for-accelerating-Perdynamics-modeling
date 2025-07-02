#include "Integrate.cuh"
//CPU
void boundary_cpu(int tt, real* y, real* disp_y, real* vel_y)
{
    for (unsigned int i = 0; i < p.N; i++)
    {
        if (y[i] < -0.025)
        {
            vel_y[i] = -50.0;
            disp_y[i] = -50.0 * tt * p.dt;
        }
        else if (y[i] > 0.025)
        {
            vel_y[i] = 50.0;
            disp_y[i] = 50.0 * tt * p.dt;
        }
    }    
}

void updata_vel_cpu(
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y)  
{
    for (int i = 0; i < p.N; i++)
    {
        acc_x[i] = pforce_x[i] / p.dens;
        acc_y[i] = pforce_y[i] / p.dens;

        vel_x[i] += acc_x[i] * p.dt;
        vel_y[i] += acc_y[i] * p.dt;

        disp_x[i] += vel_x[i] * p.dt;
        disp_y[i] += vel_y[i] * p.dt;

        if (isnan(disp_x[i]) || isnan(disp_y[i]))
        {
            printf("calculate error\n");
        }
    }
}

//GPU
__global__ void boundary_gpu(device_parameter p, int tt, real* y, real* disp_y, real* vel_y)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        if (y[i] < -0.025)
        {
            vel_y[i] = -50.0;
            disp_y[i] = -50.0 * tt * p.dt;
        }
        else if (y[i] > 0.025)
        {
            vel_y[i] = 50.0;
            disp_y[i] = 50.0 * tt * p.dt;
        }
    }
}

__global__ void updata_vel_gpu(device_parameter p, 
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        acc_x[i] = pforce_x[i] / p.dens;
        acc_y[i] = pforce_y[i] / p.dens;

        vel_x[i] += acc_x[i] * p.dt;
        vel_y[i] += acc_y[i] * p.dt;

        disp_x[i] += vel_x[i] * p.dt;
        disp_y[i] += vel_y[i] * p.dt;

        if (isnan(disp_x[i]) || isnan(disp_y[i]))
        {
            printf("calculate error\n");
        }
    }
}

//CCL
__global__ void boundary_ccl(device_parameter p, int* loop, unsigned int aero, int tt, real* y, real* disp_y, real* vel_y)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero] + loop[aero + 4]; i += stride)
    {
        if (y[i] < -0.025)
        {
            vel_y[i] = -50.0;
            disp_y[i] = -50.0 * tt * p.dt;
        }
        else if (y[i] > 0.025)
        {
            vel_y[i] = 50.0;
            disp_y[i] = 50.0 * tt * p.dt;
        }
    }
}

__global__ void updata_vel_ccl(device_parameter p, int* loop, unsigned int aero,
    real* disp_x, real* vel_x, real* acc_x, real* pforce_x,
    real* disp_y, real* vel_y, real* acc_y, real* pforce_y)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero]; i += stride)
    {
        acc_x[i] = pforce_x[i] / p.dens;
        acc_y[i] = pforce_y[i] / p.dens;

        vel_x[i] += acc_x[i] * p.dt;
        vel_y[i] += acc_y[i] * p.dt;

        disp_x[i] += vel_x[i] * p.dt;
        disp_y[i] += vel_y[i] * p.dt;

        if (isnan(disp_x[i]) || isnan(disp_y[i]))
        {
            printf("calculate error\n");
        }
    }
}
