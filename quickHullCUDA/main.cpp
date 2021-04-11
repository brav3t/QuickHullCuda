//

#include "globals.h"
#include "quickHullCPU.h"
#include "quickHULLGPU.cuh"

int main()
{
    //  01234
    //0   x
    //1   x
    //2 xxxxx
    //3   x
    //4  x x
    int h_pointsX[N10] = { 2, 2, 0, 1, 2, 3, 4, 2, 1, 3 };
    int h_pointsY[N10] = { 0, 1, 2, 2, 2, 2, 2, 3, 4, 4 };

    quickHullCPU(h_pointsX, h_pointsY, N10);

    quickHullGPU(h_pointsX, h_pointsY, N10);

    return 0;
}