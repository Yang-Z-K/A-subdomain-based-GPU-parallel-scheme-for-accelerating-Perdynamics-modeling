#include "Neighbor.cuh"
//CPU
void find_neighbor_2D(const real* x, const real* y, int* NN, int* NL)
{
    for (int i = 0; i < p.N; i++)
    {
        for (int j = i + 1; j < p.N; j++)
        {
            if (square(x[j] - x[i]) + square(y[j] - y[i]) < square(p.delta))
            {
                NL[i * p.MN + NN[i]++] = j;
                NL[j * p.MN + NN[j]++] = i;
            }
        }
    }
}

//GPU
__global__ void kernel_find_neighbor_2D(device_parameter p, 
    const real* x, const real* y, int* NN, int* NL)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride) 
    {
        for (unsigned int j = i + 1; j < p.N; j++)
        {
            if (gsquare(x[j] - x[i]) + gsquare(y[j] - y[i]) < gsquare(p.delta))
            {
                NL[i * p.MN + atomicAddInt(&NN[i], 1)] = j;
                NL[j * p.MN + atomicAddInt(&NN[j], 1)] = i;
            }
        }
    } 
}

//CCL
__global__ void ccl_find_neighbor_2D(device_parameter p, 
    const real* x, const real* y, int* NN, int* NL, int* loop, int aero)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero]; i += stride)
    {
        for (unsigned int j = i + 1; j < (loop[aero] + loop[aero + 4]); j++)
        {
            if (gsquare(x[j] - x[i]) + gsquare(y[j] - y[i]) < gsquare(p.delta))
            {
                NL[i * p.MN + atomicAddInt(&NN[i], 1)] = j;
                NL[j * p.MN + atomicAddInt(&NN[j], 1)] = i;
            }
        }
    }
}
