#include "Initial.cuh"
//CPU
static void initial_fail_cpu(int* fail)
{
    for (unsigned int i = 0; i < p.N; i++)
    {
        for (unsigned int j = 0; j < p.MN; j++)
        {
            fail[i * p.MN + j] = 1;
        }
    }
}

void base_integrate_initial_cpu(Dselect& d, BAtom& ba, BBond& bb)
{
    size_t byte = p.N * sizeof(real);

    initial_fail_cpu(bb.fail);

    memset(ba.dmg, 0, byte);

    memset(ba.disp_x, 0, byte);
    memset(ba.vel_x, 0, byte);
    memset(ba.acc_x, 0, byte);

    if (d.Dim == 2 || d.Dim == 3)
    {
        memset(ba.disp_y, 0, byte);
        memset(ba.acc_y, 0, byte);
        memset(ba.vel_y, 0, byte);

        if (d.Dim == 3)
        {
            memset(ba.disp_z, 0, byte);
            memset(ba.acc_z, 0, byte);
            memset(ba.vel_z, 0, byte);
        }
    }
}

//GPU
__global__ static void initial_fail_gpu(device_parameter p, int* fail)
{
    unsigned int instr = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = instr; i < p.N; i += stride)
    {
        for (unsigned int j = 0; j < p.MN; j++)
        {
            fail[i * p.MN + j] = 1;
        }
    }
}

void base_integrate_initial_gpu(Dselect& d, BAtom& ba, BBond& bb)
{
    size_t byte = p.N * sizeof(real);

    initial_fail_gpu << <32, 256 >> > (p, bb.fail);

    CHECK(cudaMemset(ba.dmg, 0, byte));

    CHECK(cudaMemset(ba.disp_x, 0, byte));
    CHECK(cudaMemset(ba.acc_x, 0, byte));
    CHECK(cudaMemset(ba.vel_x, 0, byte));

    if (d.Dim == 2 || d.Dim == 3)
    {
        CHECK(cudaMemset(ba.disp_y, 0, byte));
        CHECK(cudaMemset(ba.acc_y, 0, byte));
        CHECK(cudaMemset(ba.vel_y, 0, byte));

        if (d.Dim == 3)
        {
            CHECK(cudaMemset(ba.disp_z, 0, byte));
            CHECK(cudaMemset(ba.acc_z, 0, byte));
            CHECK(cudaMemset(ba.vel_z, 0, byte));
        }
    }
}

//CCL
static void ccl_initial_next(device_parameter p, int* next, int* head)
{
    for (unsigned int i = 0; i < p.N; i++)
    {
        if (i < 4) head[i] = -1;
        next[i] = -1;
    }
}

static void initial_fail_ccl(device_parameter p, int* fail)
{
    for (unsigned int i = 0; i < 4 * p.maxNum; i++)
    {
        for (unsigned int j = 0; j < p.MN; j++)
        {
            fail[i * p.MN + j] = 1;
        }
    }
}

void CCL_initial(Dselect& d, BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl)
{
    size_t byte = p.maxNum * sizeof(real);

    //region varity
    ccl_initial_next(p, ccl.next, ccl.head);
    memset(ccl.boundary, 0, ccl.boxNum * p.maxNum * sizeof(int));
    memset(ccl.loop, 0, 2 * ccl.boxNum * sizeof(int));

    //主机变量
    initial_fail_ccl(p, ccl_bb.fail);

    memset(ccl_ba.dmg, 0, ccl.boxNum * byte);

    memset(ccl_ba.disp_x, 0, ccl.boxNum * byte);
    memset(ccl_ba.vel_x, 0, ccl.boxNum * byte);
    memset(ccl_ba.acc_x, 0, ccl.boxNum * byte);

    if (d.Dim == 2 || d.Dim == 3)
    {
        memset(ccl_ba.disp_y, 0, ccl.boxNum * byte);
        memset(ccl_ba.acc_y, 0, ccl.boxNum * byte);
        memset(ccl_ba.vel_y, 0, ccl.boxNum * byte);

        if (d.Dim == 3)
        {
            memset(ccl_ba.disp_z, 0, ccl.boxNum * byte);
            memset(ccl_ba.acc_z, 0, ccl.boxNum * byte);
            memset(ccl_ba.vel_z, 0, ccl.boxNum * byte);
        }
    }

    //设备变量
    CHECK(cudaMemset(ba.disp_x, 0, byte));
    CHECK(cudaMemset(ba.acc_x, 0, byte));
    CHECK(cudaMemset(ba.vel_x, 0, byte));

    if (d.Dim == 2 || d.Dim == 3)
    {
        CHECK(cudaMemset(ba.disp_y, 0, byte));
        CHECK(cudaMemset(ba.acc_y, 0, byte));
        CHECK(cudaMemset(ba.vel_y, 0, byte));

        if (d.Dim == 3)
        {
            CHECK(cudaMemset(ba.disp_z, 0, byte));
            CHECK(cudaMemset(ba.acc_z, 0, byte));
            CHECK(cudaMemset(ba.vel_z, 0, byte));
        }
    }
}
