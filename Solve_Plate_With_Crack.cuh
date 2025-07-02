#ifndef PD_PARALLEL_SOLVE_PLATE_WITH_CRACK_CUH
#define PD_PARALLEL_SOLVE_PLATE_WITH_CRACK_CUH

#include "Base.cuh"
#include "Memory.cuh"
#include "Coord.cuh"
#include "Neighbor.cuh"
#include "Surf_Corr.cuh"
#include "Initial.cuh"
#include "Force_Bond.cuh"
#include "Integrate.cuh"
#include "Function.cuh"
#include "Save.cuh"
#include "Rigion_division.cuh"

void solve_plate_with_crack (Dselect& d);

#endif //PD_PARALLEL_SOLVE_PLATE_WITH_CRACK_CUH