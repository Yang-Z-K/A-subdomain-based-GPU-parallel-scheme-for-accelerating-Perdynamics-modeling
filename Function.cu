#include "Function.cuh"

void initialil_parameter(device_parameter& p, BaseModel_Parameter& bp)
{
    p.N = bp.N;
    p.MN = bp.MN;
    p.nx = bp.nx;
    p.ny = bp.ny;
    p.nz = bp.nz;
    p.nt = bp.nt;

    p.pi = acos(-1.0);
    p.dens = bp.dens;
    p.pratio = bp.pratio;
    p.bc = bp.bc;
    p.dx = bp.dx;
    p.delta = bp.delta;
    p.emod = bp.emod;
    p.vol = bp.vol;
    p.dt = bp.dt;
    p.mass = bp.mass;
    p.scr0 = bp.scr0;

    p.maxNum = bp.maxNum;
}

void set_crack_cpu(int* NN, int* NL, int* fail, real* x, real* y)
{
    int cnode = 0;
    for (unsigned int i = 0; i < p.N; i++)
    {
        for (unsigned int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            if (y[cnode] >= 0.0 && y[i] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
            else if (y[i] >= 0.0 && y[cnode] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
        }
    }
}

__global__ void set_crack_gpu(device_parameter p, int* NN, int* NL, int* fail, real* x, real* y)
{
    int cnode = 0;
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        for (unsigned int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];
            if (y[cnode] >= 0.0 && y[i] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
            else if (y[i] >= 0.0 && y[cnode] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
        }
    }   
}


__global__ void set_crack_ccl(device_parameter p, int* loop, unsigned int aero, int* NN, int* NL, int* fail, real* x, real* y)
{
    unsigned int cnode = 0;
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero]; i += stride)
    {
        for (unsigned int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];
            if (y[cnode] >= 0.0 && y[i] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
            else if (y[i] >= 0.0 && y[cnode] <= 0.0)
            {
                if ((fabs(x[i]) - 0.005) <= 1.0e-10 || (fabs(x[cnode]) - 0.005) <= 1.0e-10)
                {
                    fail[i * p.MN + j] = 0;
                }
            }
        }
    }
}