#include "Coord.cuh"

int coord_plate_with_crack(real* x, real* y)
{
    int num = 0;

    for (unsigned int i = 0; i < p.ny; i++)
    {
        for (unsigned int j = 0; j < p.nx; j++)
        {
            x[num] = (-(p.nx - 1) / 2.0 + j) * p.dx;
            y[num] = (-(p.ny - 1) / 2.0 + i) * p.dx;

            num++;
        }
    }

    for (unsigned int i = p.ny; i < p.ny + 3; i++)
    {
        for (unsigned int j = 0; j < p.nx; j++)
        {
            x[num] = (-(p.nx - 1) / 2.0 + j) * p.dx;
            y[num] = ((p.ny - 1) / 2.0 - i) * p.dx;

            num++;
        }
    }

    for (unsigned int i = p.ny; i < p.ny + 3; i++)
    {
        for (unsigned int j = 0; j < p.nx; j++)
        {
            x[num] = (-(p.nx - 1) / 2.0 + j) * p.dx;
            y[num] = (-(p.ny - 1) / 2.0 + i) * p.dx;

            num++;
        }
    }

    return num;
}