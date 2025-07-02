#ifndef BASE_CUH
#define BASE_CUH
#ifdef DOUBLE_PRECISION
typedef float real;
#else
typedef double real;
#endif

#include <math.h>
#include <fstream>
#include <string>
#include <iostream>
#include <iomanip>
#include "Error.cuh"

using namespace std;

inline real square(real x) { return x * x; };
inline real cubic(real x) { return x * x * x; };

inline __device__ real gsquare(real x) { return x * x; };
inline __device__ real gcubic(real x) { return x * x * x; };


//������ά������ھ���
const int MN_1D = 6;
const int MN_2D = 32;
const int MN_3D = 128;

const real pi = acos(-1.0);

//����/���У�ά�ȣ�BBPD/SBPD����̬/׼��̬
struct Dselect {
    Dselect(int device, int dim) : device(device), Dim(dim)
    {}

    int device;  
    int Dim; 
};

struct device_parameter 
{
    int N;
    int MN;
    int nx;
    int ny;
    int nz;
    int nt;

    real pi;
    real dx;
    real emod;
    real dens;
    real pratio;
    real bc;
    real delta;
    real vol;
    real dt;
    real mass;
    real scr0;

    int maxNum;
};
extern device_parameter p;

struct BaseModel_Parameter
{
    BaseModel_Parameter(const Dselect& d, real dt, real nt, real dx, real thick = 0.0,
        real length = 0.05, real height = 0.05, real width = 0.0, real emod = 192.0e9, real pratio = 1.0 / 3.0,
        real dens = 8000.0, real scr0 = 1.0, int maxNum = 10000)
        : dt(dt), nt(nt), dx(dx), thick(thick), 
        length(length), height(height), width(width), emod(emod), pratio(pratio),
        dens(dens), scr0(scr0), maxNum(maxNum)
    {
        cal_parameter(d);
    }
        
    real load;                       //����
    int nx;                    
    int ny;
    int nz;
    int N;                           //���ʵ���
    int MN;                          //�ʵ�����������

    real length;                     
    real height;                     
    real width;  
    real dt;                         //ʱ�䲽��
    real nt;                         //��ʱ�䲽
    real vol;

    real dx;                         //���Ӽ��
    real radij;                      //�����Ӽ��
    real bc;                         //������

    real sedload;                    //����Ӧ����
    real mass;                       //�ʵ�����
    real delta;                      //������Χ
    real thick;                      //ģ�ͺ��
    real area;                       //�ʵ����
    real emod;                       //����ģ��
    real pratio;                     //���ɱ�
    real dens = 8000;                //�ܶ�
    real scr0 = 0.04472;

    int maxNum;                      //CCL �������������

    void cal_parameter(const Dselect& d)
    {
        delta = 3.015 * dx;
        radij = dx / 2.0;
        sedload = 0.5 * emod / (1.0 - pratio * pratio) * 1.0e-6;
        nx = (length - dx / 2.0) / dx + 1;

        switch (d.Dim)
        {
        case 1:
            MN = MN_1D;
            area = dx * dx;
            bc = 2.0 * emod / (area * square(delta));
            N = nx;
            break;
        case 2:
            MN = MN_2D;
            ny = (height - dx / 2.0) / dx + 1;
            N = nx * (ny + 6);
            vol = dx * dx * dx;
            bc = 9.0 * emod / (pi * dx * cubic(delta));
            dt = 0.8 * pow(2.0 * dens * dx / (pi * square(delta) * dx * bc), 0.5);
            break;
        case 3:
            MN = MN_3D;
            ny = (height - dx / 2.0) / dx + 1;
            nz = (width - dx / 2.0) / dx + 1;
            N = nx * ny * nz;
            vol = dx * dx * dx;
            bc = 6.0 * emod / (pi * delta * cubic(delta) * (1.0 - 2.0 * pratio));
            break;
        default:
            break;
        }

        mass = 0.25 * dt * dt * 4.0 / 3.0 * pi * cubic(delta) * bc / dx;
    }
};

struct BAtom {
    int* NN;
    real* x;
    real* y;
    real* z;
    real* disp_x;
    real* disp_y;
    real* disp_z;
    real* vel_x;
    real* vel_y;
    real* vel_z;
    real* acc_x;
    real* acc_y;
    real* acc_z;
    real* pforce_x;
    real* pforce_y;
    real* pforce_z;

    real* fncst_x;
    real* fncst_y;
    real* fncst_z;

    real* dmg;
};

struct BBond {
    int* NL;
    int* fail;
};

struct Cell_Linkedlist{
    int* head;
    int* next;
    int* loop;
    int* boundary;

    int boxNum = 4;
};

//(����)ԭ�Ӽӷ�
static __device__ int atomicAddInt(int* address, int val)
{
    int old = *address, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, assumed + val);
    } while (assumed != old);

    return old;
}

#endif //PD_PARALLEL_BASE_CUH