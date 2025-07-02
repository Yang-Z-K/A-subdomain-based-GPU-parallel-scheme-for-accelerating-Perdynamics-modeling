#include "Force_Bond.cuh"
//CPU
void bond_force_cpu(
    real* x, real* disp_x, real* fncst_x, real* pforce_x, 
    real* y, real* disp_y, real* fncst_y, real* pforce_y, 
    int* NN, int* NL, int* fail, real* dmg)
{
    int cnode = 0;
    real nlength = 0.0, idist = 0.0;
    real fac = 0.0, theta = 0.0;
    real scx = 0.0, scy = 0.0, scr = 0.0;
    real dmgpar1 = 0.0, dmgpar2 = 0.0;

    for (int i = 0; i < p.N; i++)
    {
        pforce_x[i] = 0.0, pforce_y[i] = 0.0;
        dmgpar1 = 0.0, dmgpar2 = 0.0;

        for (int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            idist = pow(square(x[cnode] - x[i]) + square(y[cnode] - y[i]), 0.5);
            nlength = pow(square(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                square(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);
            
            //体积修正
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

            //表面修正
            if (fabs(y[cnode] - y[i]) <= 1.0e-10)
            {
                theta = 0.0;
            }
            else if (fabs(x[cnode] - x[i]) <= 1e-10)
            {
                theta = 90 * p.pi / 180.0;
            }
            else
            {
                theta = atan(fabs(y[cnode] - y[i]) / fabs(x[cnode] - x[i]));
            }

            //计算表面修正因子
            scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
            scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
            scr = pow(1.0 / ((pow(cos(theta), 2) / pow(scx, 2)) + (pow(sin(theta), 2) / pow(scy, 2))), 0.5);

            if (fail[i * p.MN + j] == 1)
            {
                pforce_x[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) / nlength;
                pforce_y[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (y[cnode] + disp_y[cnode] - y[i] - disp_y[i]) / nlength;
            }
            else
            {
                pforce_x[i] += 0.0;
                pforce_y[i] += 0.0;
            }

            if (fabs((nlength - idist) / idist) > p.scr0)
            {
                if (fabs(y[i]) <= (p.ny / 4.0) * p.dx)
                {
                    fail[i * p.MN + j] = 0;
                }
            }

            dmgpar1 += fail[i * p.MN + j] * p.vol * fac;
            dmgpar2 += p.vol * fac;

            if (isnan(pforce_x[i]) || isnan(pforce_y[i]))
            {
                cout << "pforce error" << endl;
            }
        }

        dmg[i] = 1.0 - dmgpar1 / dmgpar2;
    }
}

//GPU
__global__ void bond_force_gpu(device_parameter p,
    real* x, real* disp_x, real* fncst_x, real* pforce_x,
    real* y, real* disp_y, real* fncst_y, real* pforce_y,
    int* NN, int* NL, int* fail, real* dmg)
{
    int cnode = 0;
    real nlength = 0.0, idist = 0.0;
    real fac = 0.0, theta = 0.0;
    real scx = 0.0, scy = 0.0, scr = 0.0;
    real dmgpar1 = 0.0, dmgpar2 = 0.0;

    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        pforce_x[i] = 0.0; pforce_y[i] = 0.0;
        dmgpar1 = 0.0; dmgpar2 = 0.0;

        for (int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            idist = pow(gsquare(x[cnode] - x[i]) + gsquare(y[cnode] - y[i]), 0.5);
            nlength = pow(gsquare(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                gsquare(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);

            //体积修正
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

            //表面修正
            if (fabs(y[cnode] - y[i]) <= 1.0e-10)
            {
                theta = 0.0;
            }
            else if (fabs(x[cnode] - x[i]) <= 1e-10)
            {
                theta = 90 * p.pi / 180.0;
            }
            else
            {
                theta = atan(fabs(y[cnode] - y[i]) / fabs(x[cnode] - x[i]));
            }

            //计算表面修正因子
            scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
            scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
            scr = pow(1.0 / ((pow(cos(theta), 2) / pow(scx, 2)) + (pow(sin(theta), 2) / pow(scy, 2))), 0.5);

            if (fail[i * p.MN + j] == 1)
            {
                pforce_x[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) / nlength;
                pforce_y[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (y[cnode] + disp_y[cnode] - y[i] - disp_y[i]) / nlength;
            }
            else
            {
                pforce_x[i] += 0.0;
                pforce_y[i] += 0.0;
            }

            if (fabs((nlength - idist) / idist) > p.scr0)
            {
                if (fabs(y[i]) <= (p.ny / 4.0) * p.dx)
                {
                    fail[i * p.MN + j] = 0;
                }
            }

            dmgpar1 += fail[i * p.MN + j] * p.vol * fac;
            dmgpar2 += p.vol * fac;

            if (isnan(pforce_x[i]) || isnan(pforce_y[i]))
            {
                printf("pforce error\n");
            }
        }

        dmg[i] = 1.0 - dmgpar1 / dmgpar2;
    }
}

//CCL
__global__ void bond_force_ccl(device_parameter p, int* loop, unsigned int aero,
    real* x, real* disp_x, real* fncst_x, real* pforce_x,
    real* y, real* disp_y, real* fncst_y, real* pforce_y,
    int* NN, int* NL, int* fail, real* dmg)
{
    int cnode = 0;
    real nlength = 0.0, idist = 0.0;
    real fac = 0.0, theta = 0.0;
    real scx = 0.0, scy = 0.0, scr = 0.0;
    real dmgpar1 = 0.0, dmgpar2 = 0.0;

    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < loop[aero]; i += stride)
    {
        pforce_x[i] = 0.0; pforce_y[i] = 0.0;
        dmgpar1 = 0.0; dmgpar2 = 0.0;

        for (int j = 0; j < NN[i]; j++)
        {
            cnode = NL[i * p.MN + j];

            idist = pow(gsquare(x[cnode] - x[i]) + gsquare(y[cnode] - y[i]), 0.5);
            nlength = pow(gsquare(x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) +
                gsquare(y[cnode] + disp_y[cnode] - y[i] - disp_y[i]), 0.5);

            //体积修正
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

            //表面修正
            if (fabs(y[cnode] - y[i]) <= 1.0e-10)
            {
                theta = 0.0;
            }
            else if (fabs(x[cnode] - x[i]) <= 1e-10)
            {
                theta = 90 * p.pi / 180.0;
            }
            else
            {
                theta = atan(fabs(y[cnode] - y[i]) / fabs(x[cnode] - x[i]));
            }

            //计算表面修正因子
            scx = (fncst_x[i] + fncst_x[cnode]) / 2.0;
            scy = (fncst_y[i] + fncst_y[cnode]) / 2.0;
            scr = pow(1.0 / ((pow(cos(theta), 2) / pow(scx, 2)) + (pow(sin(theta), 2) / pow(scy, 2))), 0.5);

            if (fail[i * p.MN + j] == 1)
            {
                pforce_x[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (x[cnode] + disp_x[cnode] - x[i] - disp_x[i]) / nlength;
                pforce_y[i] += p.bc * ((nlength - idist) / idist) * p.vol * scr * fac * (y[cnode] + disp_y[cnode] - y[i] - disp_y[i]) / nlength;
            }
            else
            {
                pforce_x[i] += 0.0;
                pforce_y[i] += 0.0;
            }

            if (fabs((nlength - idist) / idist) > p.scr0)
            {
                if (fabs(y[i]) <= (p.ny / 4.0) * p.dx)
                {
                    fail[i * p.MN + j] = 0;
                }
            }

            dmgpar1 += fail[i * p.MN + j] * p.vol * fac;
            dmgpar2 += p.vol * fac;

            if (isnan(pforce_x[i]) || isnan(pforce_y[i]))
            {
                printf("pforce error\n");
            }
        }

        dmg[i] = 1.0 - dmgpar1 / dmgpar2;
    }
}