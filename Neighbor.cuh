#ifndef PD_PARALLEL_NEIGHBOR_CUH
#define PD_PARALLEL_NEIGHBOR_CUH

#include "Base.cuh"

void find_neighbor_2D(const real* x, const real* y, int* NN, int* NL);

__global__ void kernel_find_neighbor_2D(device_parameter p,
	const real* x, const real* y, int* NN, int* NL);

__global__ void ccl_find_neighbor_2D(device_parameter p,
	const real* x, const real* y, int* NN, int* NL, int* loop, int aero);

#endif