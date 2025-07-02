#include "Save.cuh"
//CPU
void save_cpu(real* x, real* y, BAtom& ba, const string FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);

    ofs << "Title = 'particle information'" << endl;
    ofs << "VARIABLES=x, y, x1, y1, dips_x, disp_y, dmage" << endl;
    ofs << "Zone i=500, j=500, f=point" << endl;

    for (int i = 0; i < p.N; i++)
    {
        if (fabs(y[i]) <= p.dx * (p.ny / 2))
        {
            ofs << std::fixed << std::setprecision(9)
                << x[i] << "\t" << y[i] << "\t"                                                        //参考构型
                << x[i] + ba.disp_x[i] << "\t" << y[i] + ba.disp_y[i] << "\t"                          //当前构型
                << ba.disp_x[i] << "\t" << ba.disp_y[i] << "\t" 
                << ba. dmg[i] << endl;
        }
    }
}

//GPU
void save_gpu(real* x, real* y, BAtom& ba, char* FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);

    ofs << "Title = 'particle information'" << endl;
    ofs << "VARIABLES=x, y, x1, y1, dips_x, disp_y, dmage, NN" << endl;
    ofs << "Zone i=500, j=506, f=point" << endl;

    real* disp_x = (real*)malloc(p.N * sizeof(real));
    real* disp_y = (real*)malloc(p.N * sizeof(real));
    real* dmg = (real*)malloc(p.N * sizeof(real));
    int* NN = (int*)malloc(p.N * sizeof(int));

    CHECK(cudaMemcpy(disp_x, ba.disp_x, p.N * sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(disp_y, ba.disp_y, p.N * sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dmg, ba.dmg, p.N * sizeof(real), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(NN, ba.NN, p.N * sizeof(int), cudaMemcpyDeviceToHost));


    for (int i = 0; i < p.N; i++)
    {
        if (fabs(y[i]) <= p.dx * (p.ny / 2))
        {
            ofs << std::fixed << std::setprecision(9)
                << x[i] << "\t" << y[i] << "\t"                                                        //参考构型
                << x[i] + disp_x[i] << "\t" << y[i] + disp_y[i] << "\t"                          //当前构型
                << disp_x[i] << "\t" << disp_y[i] << "\t"
                << dmg[i] << endl;
        }
    }

    free(disp_x);
    free(disp_y);
    free(dmg);
}

//CCL
void save_ccl(real* x, real* y, BAtom& ccl_ba, Cell_Linkedlist& ccl, char* FILE)
{
    ofstream ofs;
    ofs.open(FILE, ios::out);

    ofs << "Title = 'particle information'" << endl;
    ofs << "VARIABLES= x, y, x1, y1, disp_x, disp_y, dmg" << endl;
    ofs << "Zone i=500, j=500, f=point" << endl;

    real* disp_x = (real*)malloc(p.N * sizeof(real));
    real* disp_y = (real*)malloc(p.N * sizeof(real));
    real* dmg = (real*)malloc(p.N * sizeof(real));

    for (unsigned int aero = 0; aero < 4; aero++)
    {
        region_new_to_old(aero, ccl_ba.disp_x, ccl.head, ccl.next, disp_x);
        region_new_to_old(aero, ccl_ba.disp_y, ccl.head, ccl.next, disp_y);
        region_new_to_old(aero, ccl_ba.dmg, ccl.head, ccl.next, dmg);
    }

    for (int i = 0; i < p.N; i++)
    {
        if (fabs(y[i]) <= p.dx * (p.ny / 2))
        {
            ofs << std::fixed << std::setprecision(9)
                << x[i] << "\t" << y[i] << "\t"                                                         //参考构型
                << x[i] + disp_x[i] << "\t" << y[i] + disp_y[i] << "\t"                                 //当前构型
                << disp_x[i] << "\t" << disp_y[i] << "\t"
                << dmg[i] << endl;                                                                      //位移
        }
    }

    free(disp_x);
    free(disp_y);
    free(dmg);
}