#ifndef PD_PARALLEL_INITIAL_CUH
#define PD_PARALLEL_INITIAL_CUH
#include "Base.cuh"

//CPU��ʼ��
void base_integrate_initial_cpu(Dselect& d, BAtom& ba, BBond& bb);

void initial_cylinder_cpu(real* vel, real* disp, real* acc, real* force);

//GPU��ʼ��
void base_integrate_initial_gpu(Dselect& d, BAtom& ba, BBond& bb);

void static_initial_bond_gpu(Dselect& d, BAtom& ba, BBond& bb);

//CCL��ʼ��
void CCL_initial(Dselect& d, BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl);

#endif //PD_PARALLEL_INITIAL_CUH