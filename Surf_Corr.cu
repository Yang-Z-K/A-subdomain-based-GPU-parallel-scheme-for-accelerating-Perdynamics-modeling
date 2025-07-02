#include "Surf_Corr.cuh"
//CPU
static void Disp(real* coord, real* disp_cal, real* disp_initial_1 = NULL, real* disp_initial_2 = NULL)
{
    for (int i = 0; i < p.N; i++)
    {
        disp_cal[i] = 0.001 * coord[i];
        if (disp_initial_1 != NULL)
            disp_initial_1[i] = 0.0;
        if (disp_initial_2 != NULL)
            disp_initial_2[i] = 0.0;
    }
}

static void surface_F(real sedload_Cal, int* NN, int* NL, real* x, real* disp_x, real* fncst_Cal, real* y, real* disp_y)
{
    int  cnode = 0;
    real idist = 0.0, nlength = 0.0, fac = 0.0;
    real stendens_Cal = 0.0;

    for (int i = 0; i < p.N; i++)
    {
        stendens_Cal = 0.0;
        for (int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            idist = pow(square(x[cnode] - x[i]) + square(y[cnode] - y[i]), 0.5);

            nlength = pow(square(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                square(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);

            if (idist <= p.delta - p.dx / 2.0)
            {
                fac = 1.0;
            }
            else if (idist <= p.delta + p.dx / 2.0)
            {
                fac = (p.delta + p.dx / 2.0 - idist) / p.dx;
            }
            else
            {
                fac = 0.0;
            }

            stendens_Cal += 0.25 * p.bc * square(nlength - idist) / idist * p.vol * fac;
        }
        fncst_Cal[i] = sedload_Cal / stendens_Cal;
    }
}

void surface_correct_cpu(Dselect& d, BAtom& ba, BBond& bb)
{
    real sedload_Cal = 9.0 / 16.0 * p.emod * 1.0e-6;

    Disp(ba.x, ba.disp_x, ba.disp_y);

    surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_x, ba.y, ba.disp_y);

    Disp(ba.y, ba.disp_y, ba.disp_x);

    surface_F(sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_y, ba.y, ba.disp_y);
}

//GPU
static __global__ void kernel_Disp(device_parameter p,
    real* coord, real* disp_cal, real* disp_initial_1 = NULL, real* disp_initial_2 = NULL)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        disp_cal[i] = 0.001 * coord[i];
        if (disp_initial_1 != NULL)
            disp_initial_1[i] = 0.0;
        if (disp_initial_2 != NULL)
            disp_initial_2[i] = 0.0;
    }
}


static __global__ void kernel_surface_F(device_parameter p,
    real sedload_Cal, int* NN, int* NL, real* x, real* disp_x, real* fncst_Cal, real* y, real* disp_y)
{
    int  cnode = 0;
    real idist = 0.0, nlength = 0.0, fac = 0.0;
    real stendens;

    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        stendens = 0.0;
        for (unsigned int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];
           
            idist = pow(gsquare(x[cnode] - x[i]) + gsquare(y[cnode] - y[i]), 0.5);

            nlength = pow(gsquare(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                gsquare(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);

            if (idist <= p.delta - p.dx / 2.0)
            {
                fac = 1.0;
            }
            else if (idist <= p.delta + p.dx / 2.0)
            {
                fac = (p.delta + p.dx / 2.0 - idist) / p.dx;
            }
            else
            {
                fac = 0.0;
            }
            
            stendens += 0.25 * p.bc * (nlength - idist) * (nlength - idist) / idist * p.vol * fac;
        }

        fncst_Cal[i] = sedload_Cal / stendens;
    }
}

void surface_correct_gpu(Dselect& d, BAtom& ba, BBond& bb)
{
        real sedload_Cal = 9.0 / 16.0 * p.emod * 1.0e-6;

        //-x
        kernel_Disp << <32, 256 >> > (p, ba.x, ba.disp_x, ba.disp_y);
        kernel_surface_F << <32, 32>> > (p,
            sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_x, ba.y, ba.disp_y);

        //-y
        kernel_Disp << <32, 256 >> > (p, ba.y, ba.disp_y, ba.disp_x);
        kernel_surface_F << <32, 32>> > (p,
            sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_y, ba.y, ba.disp_y);
}

//CCL
static __global__ void ccl_kernel_Disp(device_parameter p, int* loop, unsigned int aero,
    real* coord, real* disp_cal, real* disp_initial_1 = NULL, real* disp_initial_2 = NULL)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero] + loop[aero + 4]; i += stride)
    {
        disp_cal[i] = 0.001 * coord[i];
        if (disp_initial_1 != NULL)
            disp_initial_1[i] = 0.0;
        if (disp_initial_2 != NULL)
            disp_initial_2[i] = 0.0;
    }
}

static __global__ void ccl_kernel_surface_F(device_parameter p, int* loop, unsigned int aero,
    real sedload_Cal, int* NN, int* NL, real* x, real* disp_x, real* fncst_Cal, real* y = NULL, real* disp_y = NULL)
{
    int  cnode = 0;
    real idist = 0.0, nlength = 0.0, fac = 0.0;
    real stendens;

    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero]; i += stride)
    {
        stendens = 0.0;
        for (unsigned int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            idist = pow(gsquare(x[cnode] - x[i]) + gsquare(y[cnode] - y[i]), 0.5);

            nlength = pow(gsquare(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                gsquare(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);

            if (idist <= p.delta - p.dx / 2.0)
            {
                fac = 1.0;
            }
            else if (idist <= p.delta + p.dx / 2.0)
            {
                fac = (p.delta + p.dx / 2.0 - idist) / p.dx;
            }
            else
            {
                fac = 0.0;
            }

            stendens += 0.25 * p.bc * (nlength - idist) * (nlength - idist) / idist * p.vol * fac;
        }

        fncst_Cal[i] = sedload_Cal / stendens;
    }
}

void ccl_surface_correct_gpu(Dselect& d, BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, int* loop)
{
    real sedload_Cal = 9.0 / 16.0 * p.emod * 1.0e-6;

    for (unsigned int aero = 0; aero < 4; aero++)
    {
        surface_host_to_device1(ba, ccl_ba, bb, ccl_bb, ccl, aero);

        //-x
        ccl_kernel_Disp << <32, 256 >> > (p, loop, aero, ba.x, ba.disp_x, ba.disp_y);
        ccl_kernel_surface_F << <32, 32 >> > (p, loop, aero,
            sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_x, ba.y, ba.disp_y);

        //-y
        ccl_kernel_Disp << <32, 32 >> > (p, loop, aero, ba.y, ba.disp_y, ba.disp_x);
        ccl_kernel_surface_F << <32, 32 >> > (p, loop, aero,
            sedload_Cal, ba.NN, bb.NL, ba.x, ba.disp_x, ba.fncst_y, ba.y, ba.disp_y);

        CHECK(cudaDeviceSynchronize());
        surface_device_to_host1(ba, ccl_ba, bb, ccl_bb, ccl, aero);
    }
}




