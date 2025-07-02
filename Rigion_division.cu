#include "Rigion_division.cuh"
void rigion_division(real* x, real* y, int* head, int* next, int* loop, int* boundary, int boxNum)
{
	int ix = 0, iy = 0;

	for (unsigned int i = 0; i < p.N; i++)
	{
		for (unsigned int j = 0; j < boxNum; j++)
		{
			ix = j % 2; iy = j / 2;

			if (x[i] > (ix - 1) * 0.025 - p.delta && x[i] < (ix) * 0.025 + p.delta && y[i] > (iy - 1) * 0.0253 - p.delta && y[i] < (iy) * 0.0253 + p.delta)
			{
				if (x[i] > (ix - 1) * 0.025 && x[i] < (ix) * 0.025 && y[i] > (iy - 1) * 0.0253 && y[i] < (iy) * 0.0253)
				{
					next[i] = head[j];
					head[j] = i;
					loop[j] += 1;
				}
				else
				{
					boundary[j * p.maxNum + loop[j + boxNum]] = i;
					loop[j + boxNum] += 1;
				}
			}
		}
	}
}


//区域分配(old --> new)
void region_old_to_new(int i, real* B, int* head, int* next, int* boundary, int* loop, real* A)
{
	//有效粒子
	int k = 1, t = head[i];

	B[i * p.maxNum] = A[t];

	while (next[t] != -1)
	{
		B[i * p.maxNum + k] = A[next[t]];
		k++;
		t = next[t];
	}

	//边界粒子
	for (int j = 0; j < loop[i + 4]; j++)
	{
		B[i * p.maxNum + k] = A[boundary[i * p.maxNum + j]];
		k++;
	}
}


//new --> old
void region_new_to_old(int i, real* B, int* head, int* next, real* A)
{
	//有效粒子
	int k = 1, t = head[i];
	A[t] = B[i * p.maxNum];
	while (next[t] != -1)
	{
		A[next[t]] = B[i * p.maxNum + k];
		k++;
		t = next[t];
	}
}


//数据传输
void boundary_data_update(real* new_data, Cell_Linkedlist ccl)
{
	real* temp = (real*)malloc(p.N * sizeof(real));

	for (unsigned int aero = 0; aero < 4; aero++)
		region_new_to_old(aero, new_data, ccl.head, ccl.next, temp);

	for (unsigned int aero = 0; aero < 4; aero++)
		region_old_to_new(aero, new_data, ccl.head, ccl.next, ccl.boundary, ccl.loop, temp);

	free(temp);
}

void host_to_device1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ba.x, ccl_ba.x + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.y, ccl_ba.y + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(bb.fail, ccl_bb.fail + aero * p.maxNum * p.MN, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyHostToDevice));
}

void host_to_device2(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ba.disp_x, ccl_ba.disp_x + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.disp_y, ccl_ba.disp_y + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.vel_x, ccl_ba.vel_x + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.vel_y, ccl_ba.vel_y + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.fncst_x, ccl_ba.fncst_x + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.fncst_y, ccl_ba.fncst_y + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.NN, ccl_ba.NN + aero * p.maxNum, ccl.loop[aero] * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.dmg, ccl_ba.dmg + aero * p.maxNum, ccl.loop[aero] * sizeof(real), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(bb.NL, ccl_bb.NL + aero * p.maxNum * p.MN, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyHostToDevice));
}

void host_to_device3(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ba.disp_x, ccl_ba.disp_x + aero * p.maxNum, ccl.loop[aero] * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.disp_y, ccl_ba.disp_y + aero * p.maxNum, ccl.loop[aero] * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.pforce_x, ccl_ba.pforce_x + aero * p.maxNum, ccl.loop[aero] * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.pforce_y, ccl_ba.pforce_y + aero * p.maxNum, ccl.loop[aero] * sizeof(real), cudaMemcpyHostToDevice));
}

void device_to_host1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ccl_ba.NN + aero * p.maxNum, ba.NN, ccl.loop[aero] * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_bb.NL + aero * p.maxNum * p.MN, bb.NL, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyDeviceToHost));

	CHECK(cudaMemcpy(ccl_bb.fail + aero * p.maxNum * p.MN, bb.fail, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyDeviceToHost));
}

void device_to_host2(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ccl_ba.disp_x + aero * p.maxNum, ba.disp_x, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_ba.disp_y + aero * p.maxNum, ba.disp_y, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_ba.vel_x + aero * p.maxNum, ba.vel_x, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_ba.vel_y + aero * p.maxNum, ba.vel_y, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_ba.dmg + aero * p.maxNum, ba.dmg, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));

	CHECK(cudaMemcpy(ccl_bb.fail + aero * p.maxNum * p.MN, bb.fail, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyDeviceToHost));

}

//表面修正所需的数据传输
void surface_host_to_device1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ba.x, ccl_ba.x + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(ba.y, ccl_ba.y + aero * p.maxNum, (ccl.loop[aero] + ccl.loop[aero + 4]) * sizeof(real), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(ba.NN, ccl_ba.NN + aero * p.maxNum, ccl.loop[aero] * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(bb.NL, ccl_bb.NL + aero * p.maxNum * p.MN, ccl.loop[aero] * p.MN * sizeof(int), cudaMemcpyHostToDevice));
}

void surface_device_to_host1(BAtom& ba, BAtom& ccl_ba, BBond& bb, BBond& ccl_bb, Cell_Linkedlist& ccl, unsigned int aero)
{
	CHECK(cudaMemcpy(ccl_ba.fncst_x + aero * p.maxNum, ba.fncst_x, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(ccl_ba.fncst_y + aero * p.maxNum, ba.fncst_y, ccl.loop[aero] * sizeof(real), cudaMemcpyDeviceToHost));
}