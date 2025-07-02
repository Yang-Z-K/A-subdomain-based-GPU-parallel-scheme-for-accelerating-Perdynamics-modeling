cmake_minimum_required(VERSION 3.16)
project(impact CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(Start main.cu Base.cuh 
        Error.cuh       
        Coord.cu Coord.cuh 
        Force_Bond.cu Force_Bond.cuh
        Function.cu Function.cuh 
        Initial.cu Initial.cuh 
        Integrate.cu Integrate.cuh 
        Memory.cu Memory.cuh 
        Neighbor.cu Neighbor.cuh 
        Rigion_division.cu  Rigion_division.cuh
        Save.cu Save.cuh 
        Solve_Plate_With_Crack.cu Solve_Plate_With_Crack.cuh
        Surf_Corr.cu Surf_Corr.cuh)

set_target_properties(Start PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)