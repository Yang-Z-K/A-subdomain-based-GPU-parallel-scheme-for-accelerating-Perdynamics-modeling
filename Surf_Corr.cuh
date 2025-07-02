#ifndef PD_PARALLEL_SURF_CORR_CUH
#define PD_PARALLEL_SURF_CORR_CUH

#include "Base.cuh"
#include "Rigion_division.cuh"

//CPU
void surface_correct_cpu(Dselect& d, BAtom& ba, BBond& bb);

//GPU
void surface_correct_gpu(Dselect& d, BAtom& ba, BBond& bb);

//CCL
void ccl_surface_correct_gpu(Dselect& d, BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, int* loop);

#endif //PD_PARALLEL_SURF_CORR_CUH