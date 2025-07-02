#include "Solve_Plate_With_Crack.cuh"

void solve_plate_with_crack(Dselect& d)
{   
    clock_t end1, end2;
    real timeforneighbor = 0.0, timeforsurface = 0.0, timeforinitial = 0.0, timeforsetcrack = 0.0, timeforboundary = 0.0, timefordisp = 0.0, timeforforce = 0.0, timefortransmit = 0.0;

    BAtom ba, ccl_ba;
    BBond bb, ccl_bb;
    Cell_Linkedlist ccl;

    BaseModel_Parameter bp(d, 0.0, 1250, 0.0001, 0.0001, 0.05, 0.05, 0.0, 192.0e9, 1.0/3.0, 8000, 0.04472, 64768);

    initialil_parameter(p, bp);

    real* x = new real[p.N];
    real* y = new real[p.N];
    p.N = coord_plate_with_crack(x, y);
    cout << "particle number: " << p.N << endl;
    cout << "nx: " << p.nx << " ny: " << p.ny << endl;

    int blocksize1 = 1024;
    int blocksize2 = 512;
    int gridsize1 = (p.N + 6 * p.nx - 1) / blocksize1 + 1;
    int gridsize2 = (p.N + 6 * p.nx - 1) / blocksize2 + 1;

    if (d.device != 3)
    {
        Base_Allocate(ba, bb, d);
    }

    if (d.device == 1)
    {
        CHECK(cudaMemcpy(ba.x, x, p.N * sizeof(real), cudaMemcpyHostToHost));
        CHECK(cudaMemcpy(ba.y, y, p.N * sizeof(real), cudaMemcpyHostToHost));

        for (int i = 0; i < p.N; i++)
            ba.NN[i] = 0;    

        end1 = clock();
        find_neighbor_2D(ba.x, ba.y, ba.NN, bb.NL);
        end2 = clock();
        timeforneighbor = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        surface_correct_cpu(d, ba, bb);
        end2 = clock();
        timeforsurface = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        base_integrate_initial_cpu(d, ba, bb);
        end2 = clock();
        timeforinitial = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        set_crack_cpu(ba.NN, bb.NL, bb.fail, ba.x, ba.y);
        end2 = clock();
        timeforsetcrack = (real)(end2 - end1) / CLOCKS_PER_SEC;

        for (int tt = 1; tt <= p.nt; tt++)
        {
            cout << "current time step: " << tt << endl;

            end1 = clock();
            boundary_cpu(tt, ba.y, ba.disp_y, ba.vel_y);
            end2 = clock();
            timeforboundary += (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            bond_force_cpu(
                ba.x, ba.disp_x, ba.fncst_x, ba.pforce_x,
                ba.y, ba.disp_y, ba.fncst_y, ba.pforce_y,
                ba.NN, bb.NL, bb.fail, ba.dmg);
            end2 = clock();
            timeforforce += (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            updata_vel_cpu(
                ba.disp_x, ba.vel_x, ba.acc_x, ba.pforce_x,
                ba.disp_y, ba.vel_y, ba.acc_y, ba.pforce_y);
            end2 = clock();
            timefordisp += (real)(end2 - end1) / CLOCKS_PER_SEC;

            if (tt == 1 || tt == p.nt || tt == p.nt / 2)
            {
                char* filename = (char*)malloc(20 * sizeof(char));

                sprintf(filename, "CPU%d.dat", tt);

                save_cpu(x, y, ba, filename);

                free(filename);
            }
        }
    }
    else if (d.device == 2)
    {
        CHECK(cudaMemcpy(ba.x, x, p.N * sizeof(real), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(ba.y, y, p.N * sizeof(real), cudaMemcpyHostToDevice));

        CHECK(cudaMemset(ba.NN, 0, p.N * sizeof(int)));

        end1 = clock();
        kernel_find_neighbor_2D << <gridsize1, blocksize1 >> > (p,
            ba.x, ba.y, ba.NN, bb.NL);
        CHECK(cudaDeviceSynchronize());
        end2 = clock();
        timeforneighbor = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        surface_correct_gpu(d, ba, bb);
        CHECK(cudaDeviceSynchronize());
        end2 = clock();
        timeforsurface = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        base_integrate_initial_gpu(d, ba, bb);
        CHECK(cudaDeviceSynchronize());
        end2 = clock();
        timeforinitial = (real)(end2 - end1) / CLOCKS_PER_SEC;

        end1 = clock();
        set_crack_gpu<<<gridsize1, blocksize1 >>>(p, ba.NN, bb.NL, bb.fail, ba.x, ba.y);
        CHECK(cudaDeviceSynchronize());
        end2 = clock();
        timeforsetcrack = (real)(end2 - end1) / CLOCKS_PER_SEC;

        for (int tt = 1; tt <= p.nt; tt++)
        {
            cout << "the current time step is: " << tt << endl;

            end1 = clock();
            boundary_gpu << <gridsize1, blocksize1 >> > (p, tt, ba.y, ba.disp_y, ba.vel_y);
            CHECK(cudaDeviceSynchronize());
            end2 = clock();
            timeforboundary += (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            bond_force_gpu<<<gridsize2, blocksize2 >>>(p,
                ba.x, ba.disp_x, ba.fncst_x, ba.pforce_x,
                ba.y, ba.disp_y, ba.fncst_y, ba.pforce_y,
                ba.NN, bb.NL, bb.fail, ba.dmg);
            CHECK(cudaDeviceSynchronize());
            end2 = clock();
            timeforforce += (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            updata_vel_gpu << <gridsize1, blocksize1 >> > (p,
                ba.disp_x, ba.vel_x, ba.acc_x, ba.pforce_x,
                ba.disp_y, ba.vel_y, ba.acc_y, ba.pforce_y);
            CHECK(cudaDeviceSynchronize());
            end2 = clock();
            timefordisp += (real)(end2 - end1) / CLOCKS_PER_SEC;

            if (tt == 1 || tt == p.nt)
            {
                char* filename = (char*)malloc(20 * sizeof(char));

                sprintf(filename, "GPU_%d.dat", tt);

                save_gpu(x, y, ba, filename);

                free(filename);
            }
        }
    }
    else
    {
        int* d_loop;
        CHECK(cudaMalloc((void**)&d_loop, 2 * ccl.boxNum * sizeof(int)));
        CCL_Allocate(ba, ccl_ba, bb, ccl_bb, ccl, d);

        end1 = clock();
        CCL_initial(d, ba, ccl_ba, bb, ccl_bb, ccl);
        CHECK(cudaDeviceSynchronize());
        end2 = clock();
        timeforinitial = (real)(end2 - end1) / CLOCKS_PER_SEC;

        rigion_division(x, y, ccl.head, ccl.next, ccl.loop, ccl.boundary, ccl.boxNum);
        CHECK(cudaMemcpy(d_loop, ccl.loop, 2 * ccl.boxNum * sizeof(int), cudaMemcpyHostToDevice));
        //This step is used to adjust boxNum.
        for (unsigned int i = 0; i < ccl.boxNum; i++) printf("region %d partical num: %d\n", i, ccl.loop[i] + ccl.loop[i + ccl.boxNum]);

        for (unsigned int aero = 0; aero < ccl.boxNum; aero++)
        {
            region_old_to_new(aero, ccl_ba.x, ccl.head, ccl.next, ccl.boundary, ccl.loop, x);
            region_old_to_new(aero, ccl_ba.y, ccl.head, ccl.next, ccl.boundary, ccl.loop, y);
        }

        for (unsigned int aero = 0; aero < ccl.boxNum; aero++)
        {
            end1 = clock();
            host_to_device1(ba, ccl_ba, bb, ccl_bb, ccl, aero);
            end2 = clock();
            timefortransmit += (real)(end2 - end1) / CLOCKS_PER_SEC;

            CHECK(cudaMemset(ba.NN, 0, p.maxNum * sizeof(int)));

            end1 = clock();
            ccl_find_neighbor_2D << <gridsize1, blocksize1 >> > (p, ba.x, ba.y, ba.NN, bb.NL, d_loop, aero);
            cudaDeviceSynchronize();
            end2 = clock();
            timeforneighbor += (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            set_crack_ccl << <gridsize1, blocksize1 >> > (p, d_loop, aero, ba.NN, bb.NL, bb.fail, ba.x, ba.y);
            CHECK(cudaDeviceSynchronize());
            end2 = clock();
            timeforsetcrack = (real)(end2 - end1) / CLOCKS_PER_SEC;

            end1 = clock();
            device_to_host1(ba, ccl_ba, bb, ccl_bb, ccl, aero);
            end2 = clock();
            timefortransmit += (real)(end2 - end1) / CLOCKS_PER_SEC;
        }

        end1 = clock();
        ccl_surface_correct_gpu(d, ba, ccl_ba, bb, ccl_bb, ccl, d_loop);
        cudaDeviceSynchronize();
        end2 = clock();
        timeforsurface += (real)(end2 - end1) / CLOCKS_PER_SEC;

        boundary_data_update(ccl_ba.fncst_x, ccl);
        boundary_data_update(ccl_ba.fncst_y, ccl);

        for (unsigned int tt = 1; tt <= p.nt; tt++)
        {
            cout << "the current time step is: " << tt << endl;

            //更新边界区域的位移和速度
            boundary_data_update(ccl_ba.disp_x, ccl);
            boundary_data_update(ccl_ba.disp_y, ccl);

            for (unsigned int aero = 0; aero < ccl.boxNum; aero++)
            {
                end1 = clock();
                host_to_device1(ba, ccl_ba, bb, ccl_bb, ccl, aero);
                host_to_device2(ba, ccl_ba, bb, ccl_bb, ccl, aero);
                end2 = clock();
                timefortransmit += (real)(end2 - end1) / CLOCKS_PER_SEC;

                end1 = clock();
                boundary_ccl << <gridsize1, blocksize1 >> > (p, d_loop, aero, tt, ba.y, ba.disp_y, ba.vel_y);
                end2 = clock();
                timeforboundary += (real)(end2 - end1) / CLOCKS_PER_SEC;

                end1 = clock();
                bond_force_ccl << <gridsize2, blocksize2 >> > (p, d_loop, aero,
                    ba.x, ba.disp_x, ba.fncst_x, ba.pforce_x,
                    ba.y, ba.disp_y, ba.fncst_y, ba.pforce_y,
                    ba.NN, bb.NL, bb.fail, ba.dmg);
                CHECK(cudaDeviceSynchronize());
                end2 = clock();
                timeforforce += (real)(end2 - end1) / CLOCKS_PER_SEC;

                end1 = clock();
                updata_vel_ccl << <gridsize1, blocksize1 >> > (p, d_loop, aero,
                    ba.disp_x, ba.vel_x, ba.acc_x, ba.pforce_x,
                    ba.disp_y, ba.vel_y, ba.acc_y, ba.pforce_y);
                CHECK(cudaDeviceSynchronize());
                end2 = clock();
                timefordisp += (real)(end2 - end1) / CLOCKS_PER_SEC;

                end1 = clock();
                device_to_host2(ba, ccl_ba, bb, ccl_bb, ccl, aero);
                end2 = clock();
                timefortransmit += (real)(end2 - end1) / CLOCKS_PER_SEC;

                if (tt == 1 || tt == p.nt)
                {
                    char* filename = (char*)malloc(20 * sizeof(char));

                    sprintf(filename, "CCL_%d.dat", tt);

                    save_ccl(x, y, ccl_ba, ccl, filename);

                    free(filename);
                }
            }
        }
    }

    cout << "time for neighbor is: " << timeforneighbor << endl;
    cout << "time for surface is: " << timeforsurface << endl;
    cout << "time for initial is: " << timeforinitial << endl;
    cout << "time for setcrack is: " << timeforsetcrack << endl;
    cout << "time for boundary is: " << timeforboundary << endl;
    cout << "time for force is: " << timeforforce << endl;
    cout << "time for disp is: " << timefordisp << endl;
    if (d.device == 3)
    {
        cout << "time for transmit is: " << timefortransmit << endl;
        CCL_Free(ccl_ba, ccl_bb, ccl, d);
    }
    else
    {
        cout << "Cell-linked list is not used, and the data transfer time is negligible" << endl;
    }

    Base_Free(ba, bb, d);
}

