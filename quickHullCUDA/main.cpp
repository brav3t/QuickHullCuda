//

#include "globals.h"
#include "quickHullCPU.h"
#include "quickHULLGPU.cuh"

void pointGenerator(int* array1, int* array2);
void measureFunctionTime(std::function<void(int*, int*)> function, int* param1, int* param2);

int main()
{
    // Easy sample for result check.
    //  01234
    //0   x
    //1   x
    //2 xxxxx
    //3   x
    //4  x x
    //int pointsX[N] = { 2, 2, 0, 1, 2, 3, 4, 2, 1, 3 };
    //int pointsY[N] = { 0, 1, 2, 2, 2, 2, 2, 3, 4, 4 };
    
    int pointsX[N];
    int pointsY[N];
    pointGenerator(pointsX, pointsY);

    //for (size_t i = 0; i < N; i++)
    //    printf("(%d, %d), ", pointsX[i], pointsY[i]);
    //printf("\n\n");

    measureFunctionTime(quickHullCPU, pointsX, pointsY);
    measureFunctionTime(quickHullGPU, pointsX, pointsY);

    return 0;
}

void pointGenerator(int* array1, int* array2)
{
    srand(time(0));
    for (size_t i = 0; i < 2 * N; ++i)
        (i < N) ? array1[i] = rand() % N : array2[i - N] = rand() % N;
}

void measureFunctionTime(std::function<void(int*, int*)> function, int* param1, int* param2)
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    function(param1, param2);
    auto t2 = high_resolution_clock::now();

    /* Getting number of milliseconds as an integer. */
    auto ms_int = duration_cast<milliseconds>(t2 - t1);

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli> ms_double = t2 - t1;

    std::cout << ms_int.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";
}
