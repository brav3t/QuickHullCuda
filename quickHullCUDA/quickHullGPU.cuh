//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "globals.h"

#define BLOCK_SIZE 512 // Block size should not be smaller than N -> breaks MinSearch()
#define GRID_SIZE ((N - 1) / BLOCK_SIZE + 1)

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("\nError at %s line:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE); } } while(0)

// Gets the minimum value from the array.
__global__ static void MinSearch(int* min, int* array);
__global__ static void MaxSearch(int* max, int* array);
__global__ void getIndexOfValue(int* index, int* value, int* array);
// Returns and calculates all points distances from P1-P2 line. If side is not correct then distance is zero.
__global__ void getDistancesFromLine(int* distances, int* P1Idx, int* P2Idx, int* expectedSide, int* pointsX, int* pointsY);
// Get the side of P0 point from P1-P2 line.
__global__ void getSide(int* side, int* P1Idx, int* P2Idx, int* P0Idx, int* pointsX, int* pointsY);
// If P1 or P2 is not in hull then add to it.
__global__ void tryAddToHull(int* P1Idx, int* P2Idx, int* pointsX, int* pointsY);

// Calculates distances and add element to hull if no farther element is found.
void quickHull(int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int* d_expectedSide, int* d_pointsX, int* d_pointsY);
// Main function to calcualte min and max values and start the recursion.
void quickHullGPU(int* h_pointsX, int* h_pointsY);
