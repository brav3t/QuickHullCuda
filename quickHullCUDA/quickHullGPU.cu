//

#include "quickHullGPU.cuh"

__device__ int d_minX = 0;
__device__ int d_maxX = 0;

__global__ void getMinMax(int* d_pointsX, size_t N)
{
    const int blockSize = blockIdx.x * blockDim.x;
    int globalX = blockSize + threadIdx.x;

    if (N < globalX)
        return;

    if (d_pointsX[globalX] < d_pointsX[d_minX])
        atomicExch(&d_minX, globalX);

    if (d_pointsX[d_maxX] < d_pointsX[globalX])
        atomicExch(&d_maxX, globalX);
}

void quickHullGPU(int* h_pointsX, int* h_pointsY, size_t N)
{
    if (N < 3)
        return;

    int* d_pointsX;
    int* d_pointsY;
    CUDA_CALL(cudaMalloc((void**)&d_pointsX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_pointsY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_pointsX, h_pointsX, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pointsY, h_pointsY, N * sizeof(int), cudaMemcpyHostToDevice));

    getMinMax<<<1, N>>>(d_pointsX, N);

    int minX = -1;
    int maxX = -1;
    CUDA_CALL(cudaMemcpyFromSymbol(&minX, &d_minX, sizeof(int)));
    cudaError_t err = cudaGetLastError();
    CUDA_CALL(cudaMemcpyFromSymbol(&maxX, &d_maxX, sizeof(int)));
    //printf("minX = %d, maxX = %d", minX, maxX);
    //int debug = 0;


    CUDA_CALL(cudaFree(d_pointsX));
    CUDA_CALL(cudaFree(d_pointsY));
}
