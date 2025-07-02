#ifndef PD_PARALLEL_SAVE_CUH
#define PD_PARALLEL_SAVE_CUH

#include "Base.cuh"
#include "Rigion_division.cuh"

//CPU
void save_cpu(real* x, real* y, BAtom& ba, const string FILE);

//GPU
void save_gpu(real* x, real* y, BAtom& ba, char* FILE);

//CCL
void save_ccl(real* x, real* y, BAtom& ccl_ba, Cell_Linkedlist& ccl, char* FILE);

#endif //PD_PARALLEL_SAVE_CUH