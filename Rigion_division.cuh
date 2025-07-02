#ifndef PD_PARALLEL_RIGION_DIVISION_CUH
#define PD_PARALLEL_RIGION_DIVISION_CUH

#include "Base.cuh"
#include "Function.cuh"

//void rigion_division(real* x, real* y, int* head, int* next, int* loop, int* boundary);

void rigion_division(real* x, real* y, int* head, int* next, int* loop, int* boundary, int boxNum);

void region_old_to_new(int i, real* B, int* head, int* next, int* boundary, int* loop, real* A);

void region_new_to_old(int i, real* B, int* head, int* next, real* A);

void boundary_data_update(real* new_data, Cell_Linkedlist ccl);

void host_to_device1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

void host_to_device2(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

void device_to_host1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

void device_to_host2(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

void surface_host_to_device1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

void surface_device_to_host1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero);

#endif