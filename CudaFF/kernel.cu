// IY4JXY

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <curand.h>

#include <stdio.h>
#include <stdlib.h>

#define N10 10
#define N1000 1000

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
//    printf("Error at %s:%d\n",__FILE__,__LINE__);\
//    return EXIT_FAILURE;}} while(0)

// Results - the hull - come here
int* hullX = (int*)malloc(256 * sizeof(int));
int* hullY = (int*)malloc(256 * sizeof(int));
int hullSize = -1;

//
void quickHull(int* pointsX, int* pointsY, size_t N, int x1, int y1, int x2, int y2, int side)
{
    // Get value proportional to distance from line P1(x1, y1) P2(x2, y2) to P0(x0, y0).
    static auto getDistFromLine = [](int x1, int y1, int x2, int y2, int x0, int y0)
    {
        return abs((y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1));
    };

    // Gets side of point P0(x0, y0) from line defined by P1(x1, y1) and P2(x2, y2).
    static auto getSide = [](int x1, int y1, int x2, int y2, int x0, int y0)
    {
        int area = (y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1);
        if (0 < area)
            return 1;
        if (area < 0)
            return - 1;
        return 0;
    };

    int idxMaxDist = -1;
    int maxDist = 0;

    // Get the point with max distance from line.
    for (size_t i = 0; i < N; ++i)
    {
        int dist = getDistFromLine(x1, y1, x2, y2, pointsX[i], pointsY[i]);
        if (getSide(x1, y1, x2, y2, pointsX[i], pointsY[i]) == side && maxDist < dist)
        {
            idxMaxDist = i;
            maxDist = dist;
        }
    }

    // If no point found then add the end points P1 and P2 of line to hull.
    if (idxMaxDist == -1)
    {
        // Resize hull containter if not enought space for two new points.
        static int hullArraySize = 256;
        if (hullArraySize < hullSize + 2)
        {
            hullArraySize *= 2;
            if (void* newHullX = realloc(hullX, hullArraySize)) // error handling
                hullX = (int*)newHullX; 
            if (void* newHullY = realloc(hullY, hullArraySize))
                hullY = (int*)newHullY;
        }

        static auto findInHull = [](int x, int y, int* hullX, int* hullY, int hullSize) 
        {
            for (int i = 0; i <= hullSize; ++i)
            {
                if (hullX[i] == x && hullY[i] == y)
                    return true;
            }
            return false;
        };

        // Add end of lines to hull. Check for duplicats.
        if (!findInHull(x1, y1, hullX, hullY, hullSize))
        {
            ++hullSize;
            hullX[hullSize] = x1;
            hullY[hullSize] = y1;
        }
        if (!findInHull(x2, y2, hullX, hullY, hullSize))
        {
            ++hullSize;
            hullX[hullSize] = x2;
            hullY[hullSize] = y2;
        }

        return;
    }

    // Check for sides divided by idxMaxDist.
    quickHull(pointsX, pointsY, N, pointsX[idxMaxDist], pointsY[idxMaxDist], x1, y1, -getSide(pointsX[idxMaxDist], pointsY[idxMaxDist], x1, y1, x2, y2));
    quickHull(pointsX, pointsY, N, pointsX[idxMaxDist], pointsY[idxMaxDist], x2, y2, -getSide(pointsX[idxMaxDist], pointsY[idxMaxDist], x2, y2, x1, y1));
}

// 
void quickHullCPU(int* pointsX, int* pointsY, size_t N)
{
    if (N < 3)
        return;

    int minX = 0;
    int maxX = 0;
    for (size_t i = 1; i < N; ++i)
    {
        if (pointsX[i] < pointsX[minX])
            minX = i;

        if (pointsX[maxX] < pointsX[i])
            maxX = i;
    }

    // Run for both sides of the minX maxX defiend line.
    quickHull(pointsX, pointsY, N, pointsX[minX], pointsY[minX], pointsX[maxX], pointsY[maxX], 1);
    quickHull(pointsX, pointsY, N, pointsX[minX], pointsY[minX], pointsX[maxX], pointsY[maxX], -1);

    // Print the hull
    for (size_t i = 0; i <= hullSize; i++)
        printf("(%d, %d), ", hullX[i], hullY[i]);
}


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


    //size_t n = 10;
    //curandGenerator_t gen;
    //unsigned int* devData;
    //unsigned int* hostData;

    ///* Allocate n floats on host */
    //hostData = (unsigned*)calloc(n, sizeof(unsigned int));

    ///* Allocate n floats on device */
    //CUDA_CALL(cudaMalloc((void**)&devData, n * sizeof(unsigned int)));

    ///* Create pseudo-random number generator */
    //CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    ///* Set seed */ 
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    ///* Generate n floats on device */
    //CURAND_CALL(curandGenerate(gen, devData, n));

    ///* Copy device memory to host */
    //CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ///* Show result */
    //for (size_t i = 0; i < n; i++) {
    //    printf("%d ", hostData[i]);
    //}
    //printf("\n");

    ///* Cleanup */
    //CURAND_CALL(curandDestroyGenerator(gen));
    //CUDA_CALL(cudaFree(devData));
    //free(hostData);
    //return EXIT_SUCCESS;

    //

	//int threadX = 32;
	//int threadY = 32;
	//int threadCount = N;
	//int blockCount = (threadCount - 1) / N + 1;
	//
	//generatePoints<<<blockCount, dim3(threadX, threadY)>>>();
	//cudaError_t err = cudaSuccess;
	//err = cudaGetLastError();
	//if (err != cudaSuccess)
	//	printf("generatePoints error");

    free(hullX);
    free(hullY);

	return 0;
}
