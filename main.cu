#include "Solve_Plate_With_Crack.cuh"
device_parameter p;

int main() {
    clock_t start = clock();
    
    Dselect d{3, 2};
    //所用设备，cpu：1， gpu：2， CCL：3
    //记录模型维度，一维：1，二维：2，三维：3

    int mode = 1; 

    switch (mode)
    {
    case 1:
        solve_plate_with_crack(d);
        break;
    default:
        break;
    }

    clock_t end = clock();
    cout << "total time: " << (real)(end - start) / CLOCKS_PER_SEC << endl;

    return 0;
}


