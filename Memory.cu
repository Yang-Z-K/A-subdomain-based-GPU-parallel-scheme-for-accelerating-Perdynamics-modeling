#include "Memory.cuh"
static void base_alloc_memory_cpu(const Dselect& d, real** coord, real** disp, real** vel, real** acc, real** pforce, real** fncst)
{
    size_t real_m1 = p.N * sizeof(real);

    *coord = (real*)malloc(real_m1);
    *disp = (real*)malloc(real_m1);
    *vel = (real*)malloc(real_m1);
    *acc = (real*)malloc(real_m1);
    *pforce = (real*)malloc(real_m1);
    *fncst = (real*)malloc(real_m1);
}

void Base_Allocate_cpu(BAtom& ba, BBond& bb, const Dselect& d) 
{
    size_t real_m1 = p.N * sizeof(real);
    size_t int_m1 = p.N * sizeof(int);
    size_t int_mn = p.N * p.MN * sizeof(int);

    ba.NN = (int*)malloc(int_m1);
    ba.dmg = (real*)malloc(real_m1);

    bb.NL = (int*)malloc(int_mn);
    bb.fail = (int*)malloc(int_mn);

    base_alloc_memory_cpu(d, &ba.x, &ba.disp_x, &ba.vel_x, &ba.acc_x, &ba.pforce_x, &ba.fncst_x);
    if (d.Dim == 2 || d.Dim == 3)
    {
        base_alloc_memory_cpu(d, &ba.y, &ba.disp_y, &ba.vel_y, &ba.acc_y, &ba.pforce_y, &ba.fncst_y);
        if (d.Dim == 3)
        {
            base_alloc_memory_cpu(d, &ba.z, &ba.disp_z, &ba.vel_z, &ba.acc_z, &ba.pforce_z, &ba.fncst_z);
        }
    }
}

static void base_alloc_memory_gpu(const Dselect& d, void** coord, void** disp, void** vel, void** acc, void** pforce, void** fncst)
{
    size_t real_m1 = p.N * sizeof(real);

    CHECK(cudaMalloc(coord, real_m1));
    CHECK(cudaMalloc(disp, real_m1));
    CHECK(cudaMalloc(vel, real_m1));
    CHECK(cudaMalloc(acc, real_m1));
    CHECK(cudaMalloc(pforce, real_m1));
    CHECK(cudaMalloc(fncst, real_m1));
}


void Base_Allocate_gpu(BAtom& ba, BBond& bb, const Dselect& d) 
{
    size_t real_m1 = p.N * sizeof(real);
    size_t int_m1 = p.N * sizeof(int);
    size_t int_mn = p.N * p.MN * sizeof(int);

    CHECK(cudaMalloc((void**)&ba.NN, int_m1));
    CHECK(cudaMalloc((void**)&ba.dmg, real_m1));

    CHECK(cudaMalloc((void**)&bb.NL, int_mn));
    CHECK(cudaMalloc((void**)&bb.fail, int_mn));

    base_alloc_memory_gpu(d, (void**)&ba.x, (void**)&ba.disp_x, (void**)&ba.vel_x, (void**)&ba.acc_x, (void**)&ba.pforce_x, (void**)&ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        base_alloc_memory_gpu(d, (void**)&ba.y, (void**)&ba.disp_y, (void**)&ba.vel_y, (void**)&ba.acc_y, (void**)&ba.pforce_y, (void**)&ba.fncst_y);

        if (d.Dim == 3)
        {
            base_alloc_memory_gpu(d, (void**)&ba.z, (void**)&ba.disp_z, (void**)&ba.vel_z, (void**)&ba.acc_z, (void**)&ba.pforce_z, (void**)&ba.fncst_z);
        }
    }
}


void Base_Allocate(BAtom& ba, BBond& bb, const Dselect& d)
{
    if (d.device == 1)
    {
        Base_Allocate_cpu(ba, bb, d);
    }
    else if (d.device == 2)
    {
        Base_Allocate_gpu(ba, bb, d);
    }
}


static void base_free_memory_cpu(
    const Dselect& d, real* coord, real* disp, real* vel, real* acc, real* pforce, real* fncst)
{
    free(coord);
    free(disp);
    free(vel);
    free(acc);
    free(pforce);
    free(fncst);
}

void base_free_cpu(BAtom& ba, BBond& bb, const Dselect& d)
{
    free(ba.NN);
    free(ba.dmg);

    free(bb.NL);
    free(bb.fail);

    base_free_memory_cpu(d, ba.x, ba.disp_x, ba.vel_x, ba.acc_x, ba.pforce_x, ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        base_free_memory_cpu(d, ba.y, ba.disp_y, ba.vel_y, ba.acc_y, ba.pforce_y, ba.fncst_y);

        if (d.Dim == 3)
        {
            base_free_memory_cpu(d, ba.z, ba.disp_z, ba.vel_z, ba.acc_z, ba.pforce_z, ba.fncst_z);
        }
    }
}


static void base_free_memory_gpu(
    const Dselect& d, real* coord, real* disp, real* vel, real* acc, real* pforce, real* fncst)
{
    CHECK(cudaFree(coord));
    CHECK(cudaFree(disp));
    CHECK(cudaFree(vel));
    CHECK(cudaFree(acc));
    CHECK(cudaFree(pforce));
    CHECK(cudaFree(fncst));
}


void base_free_gpu(BAtom& ba, BBond& bb, const Dselect& d)
{
    CHECK(cudaFree(ba.NN));
    CHECK(cudaFree(ba.dmg));

    CHECK(cudaFree(bb.NL));
    CHECK(cudaFree(bb.fail));

    base_free_memory_gpu(d, ba.x, ba.disp_x, ba.vel_x, ba.acc_x, ba.pforce_x, ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        base_free_memory_gpu(d, ba.y, ba.disp_y, ba.vel_y, ba.acc_y, ba.pforce_y, ba.fncst_y);

        if (d.Dim == 3)
        {
            base_free_memory_gpu(d, ba.z, ba.disp_z, ba.vel_z, ba.acc_z, ba.pforce_z, ba.fncst_z);
        }
    }
}


void Base_Free(BAtom& ba, BBond& bb, const Dselect& d)
{
    if (d.device == 1)
    {
        base_free_cpu(ba, bb, d);
    }
    else if (d.device == 2)
    {
        base_free_gpu(ba, bb, d);
    }
}


//以下内容为CCL的内存分配及释放
static void ccl_alloc_memory_cpu(const Dselect& d, int boxNum, real** coord, real** disp, real** vel, real** acc, real** pforce, real** fncst)
{
    size_t real_m1 = boxNum * p.maxNum * sizeof(real);

    *coord = (real*)malloc(real_m1);
    *disp = (real*)malloc(real_m1);
    *vel = (real*)malloc(real_m1);
    *acc = (real*)malloc(real_m1);
    *pforce = (real*)malloc(real_m1);
    *fncst = (real*)malloc(real_m1);
}

static void CCL_alloc_memory_gpu(const Dselect& d, void** coord, void** disp, void** vel, void** acc, void** pforce, void** fncst)
{
    size_t real_m1 = p.maxNum * sizeof(real);

    CHECK(cudaMalloc(coord, real_m1));
    CHECK(cudaMalloc(disp, real_m1));
    CHECK(cudaMalloc(vel, real_m1));
    CHECK(cudaMalloc(acc, real_m1));
    CHECK(cudaMalloc(pforce, real_m1));
    CHECK(cudaMalloc(fncst, real_m1));
}


void CCL_Allocate(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, const Dselect& d)
{
    size_t real_m1 = p.maxNum * sizeof(real);
    size_t int_m1 = p.maxNum * sizeof(int);
    size_t int_mn = p.maxNum * p.MN * sizeof(int);

    //主机参数
    ccl.boundary = (int*)malloc(ccl.boxNum * p.maxNum * sizeof(int));
    ccl.head = (int*)malloc(ccl.boxNum * sizeof(int));
    ccl.next = (int*)malloc(p.N * sizeof(int));
    ccl.loop = (int*)malloc(2 * ccl.boxNum * sizeof(int));

    ccl_ba.NN = (int*)malloc(ccl.boxNum * int_m1);
    ccl_ba.dmg = (real*)malloc(ccl.boxNum * real_m1);

    ccl_bb.NL = (int*)malloc(ccl.boxNum * int_mn);
    ccl_bb.fail = (int*)malloc(ccl.boxNum * int_mn);

    ccl_alloc_memory_cpu(d, ccl.boxNum, &ccl_ba.x, &ccl_ba.disp_x, &ccl_ba.vel_x, &ccl_ba.acc_x, &ccl_ba.pforce_x, &ccl_ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        ccl_alloc_memory_cpu(d, ccl.boxNum, &ccl_ba.y, &ccl_ba.disp_y, &ccl_ba.vel_y, &ccl_ba.acc_y, &ccl_ba.pforce_y, &ccl_ba.fncst_y);
        if (d.Dim == 3)
        {
            ccl_alloc_memory_cpu(d, ccl.boxNum, &ccl_ba.z, &ccl_ba.disp_z, &ccl_ba.vel_z, &ccl_ba.acc_z, &ccl_ba.pforce_z, &ccl_ba.fncst_z);
        }
    }

    //设备参数
    CHECK(cudaMalloc((void**)&ba.NN, int_m1));
    CHECK(cudaMalloc((void**)&ba.dmg, real_m1));

    CHECK(cudaMalloc((void**)&bb.NL, int_mn));
    CHECK(cudaMalloc((void**)&bb.fail, int_mn));

    CCL_alloc_memory_gpu(d, (void**)&ba.x, (void**)&ba.disp_x, (void**)&ba.vel_x, (void**)&ba.acc_x, (void**)&ba.pforce_x, (void**)&ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        CCL_alloc_memory_gpu(d, (void**)&ba.y, (void**)&ba.disp_y, (void**)&ba.vel_y, (void**)&ba.acc_y, (void**)&ba.pforce_y, (void**)&ba.fncst_y);

        if (d.Dim == 3)
        {
            CCL_alloc_memory_gpu(d, (void**)&ba.z, (void**)&ba.disp_z, (void**)&ba.vel_z, (void**)&ba.acc_z, (void**)&ba.pforce_z, (void**)&ba.fncst_z);
        }
    }
}

void CCL_Free(BAtom& ccl_ba, BBond& ccl_bb, Cell_Linkedlist& ccl, const Dselect& d)
{
    free(ccl.boundary);
    free(ccl.head);
    free(ccl.next);
    free(ccl.loop);

    free(ccl_ba.NN);
    free(ccl_ba.dmg);
    free(ccl_bb.NL);
    free(ccl_bb.fail);

    base_free_memory_cpu(d, ccl_ba.x, ccl_ba.disp_x, ccl_ba.vel_x, ccl_ba.acc_x, ccl_ba.pforce_x, ccl_ba.fncst_x);

    if (d.Dim == 2 || d.Dim == 3)
    {
        base_free_memory_cpu(d, ccl_ba.y, ccl_ba.disp_y, ccl_ba.vel_y, ccl_ba.acc_y, ccl_ba.pforce_y, ccl_ba.fncst_y);

        if (d.Dim == 3)
        {
            base_free_memory_cpu(d, ccl_ba.z, ccl_ba.disp_z, ccl_ba.vel_z, ccl_ba.acc_z, ccl_ba.pforce_z, ccl_ba.fncst_z);
        }
    }
}