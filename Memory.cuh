#ifndef PD_PARALLEL_MEMORY_CUH
#define PD_PARALLEL_MEMORY_CUH

#include "Base.cuh"

void Base_Allocate(BAtom& ba, BBond& bb, const Dselect& d);

void Base_Free(BAtom& ba, BBond& bb, const Dselect& d);

void CCL_Allocate(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, const Dselect& d);

void CCL_Free(BAtom& ccl_ba, BBond& ccl_bb, Cell_Linkedlist& ccl, const Dselect& d);

#endif //PD_PARALLEL_MEMORY_CUH