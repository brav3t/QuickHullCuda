//

#include "quickHullGPU.cuh"

const unsigned int gridSize = (N - 1) / BLOCK_SIZE + 1;

// Has warp
// Has shared memory bank conflicts
// Has instruction overhead
__global__ void reduceMin(int* min, int* array)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    extern __shared__ int sh_min[];
    // Each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    sh_min[tid] = array[gid];

    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMin(&sh_min[i], sh_min[i + s]);

        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0)
    {
        min[blockIdx.x] = sh_min[0];
        printf("min = %d\n", sh_min[tid]);
    }
}

__global__ void reduceMax(int* max, int* array)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (N <= gid)
        return;

    extern __shared__ int sh_max[];
    // Each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;

    sh_max[tid] = array[gid];

    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMax(&sh_max[i], sh_max[i + s]);

        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0)
    {
        max[blockIdx.x] = sh_max[0];
        printf("max = %d\n", sh_max[tid]);
    }
}

__global__ void getIndexOfValue(int* index, int* value, int* array)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    if (gid == 0)
        *index = -1;

    __syncthreads(); //causes warp but needed for index not get overwritten

    if (array[gid] == *value)
    {
        *index = gid;
        printf("index = %d\n", *index);
    }
}

__global__ void getDistancesFromLine(int* distances, int* P1Idx, int* P2Idx, int* pointsX, int* pointsY, int expectedSide)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    // P1
    int x1 = pointsX[*P1Idx];
    int y1 = pointsY[*P1Idx];
    // P2
    int x2 = pointsX[*P2Idx];
    int y2 = pointsY[*P2Idx];

    // P0
    int x0 = pointsX[gid];
    int y0 = pointsY[gid];

    // Get side of P0 from P1-P2 line
    int side = 0;
    int area = (y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1);
    if (0 < area)
        side = 1;
    else if (area < 0)
        side = -1;

    if (side == expectedSide)
    {
        // Distance of P0 from P1-P2 line
        int dist = abs((y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1));
        distances[gid] = dist;
    }
    else
        distances[gid] = 0;

    printf("distances[%d] = %d\n", gid, distances[gid]);
}

__global__ void tryAddToHull(int* hullX, int* hullY, int* d_hullSize, int* P1Idx, int* P2Idx, int* pointsX, int* pointsY)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ bool sh_foundP1;
    __shared__ bool sh_foundP2;

    if (!tid)
    {
        sh_foundP1 = false;
        sh_foundP2 = false;
    }

    if (hullX[gid] == pointsX[*P1Idx] && hullY[gid] == pointsY[*P1Idx])
        sh_foundP1 = true;
    
    if (hullX[gid] == pointsX[*P2Idx] && hullY[gid] == pointsY[*P2Idx])
        sh_foundP2 = true;

    __syncthreads();

    if (!tid && !sh_foundP1)
    {
        hullX[*d_hullSize] = pointsX[*P1Idx];
        hullY[*d_hullSize] = pointsY[*P1Idx];
        ++(*d_hullSize);
    }

    if (!tid && !sh_foundP2)
    {
        hullX[*d_hullSize] = pointsX[*P2Idx];
        hullY[*d_hullSize] = pointsY[*P2Idx];
        ++(*d_hullSize);
    }
}

void quickHull(int* d_hullX, int* d_hullY, int* d_hullSize, int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int* d_pointsX, int* d_pointsY, int expectedSide)
{
    int* d_distances;
    CUDA_CALL(cudaMalloc((void**)&d_distances, N * sizeof(int)));
    getDistancesFromLine<<<gridSize, BLOCK_SIZE>>>(d_distances, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY, expectedSide);

    int* d_maxDist;
    CUDA_CALL(cudaMalloc((void**)&d_maxDist, sizeof(int)));
    reduceMax<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_maxDist, d_distances);

    // Need to copy back max distance from line for CPU to decide that the recursion can be return, and the points can be added to hull.
    int h_maxDist = 0;
    CUDA_CALL(cudaMemcpy(&h_maxDist, d_maxDist, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_maxDist)
    {
        // Need to copy back hullSize for thread count.
        int h_hullSize;
        CUDA_CALL(cudaMemcpy(&h_hullSize, d_hullSize, sizeof(int), cudaMemcpyDeviceToHost));
        int blockSize = h_hullSize + 1;
        tryAddToHull<<<gridSize, blockSize>>>(d_hullX, d_hullY, d_hullSize, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY);
        return;
    }

    int* d_maxDistIdx;
    CUDA_CALL(cudaMalloc((void**)&d_maxDistIdx, sizeof(int)));
    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_maxDistIdx, d_maxDist, d_distances); // ha több max érték van bármelyik beleírhatódik, nem baj a versenyhelyzet

    CUDA_CALL(cudaFree(d_distances));

    quickHull(d_hullX, d_hullY, d_hullSize, d_P1Idx, d_maxDistIdx, d_pointsX, d_pointsY, expectedSide);
    quickHull(d_hullX, d_hullY, d_hullSize, d_maxDistIdx, d_P2Idx, d_pointsX, d_pointsY, expectedSide);
}

void quickHullGPU(int* h_pointsX, int* h_pointsY)
{
    if (N < 3 && !h_pointsX && !h_pointsY)
        return;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Copy coordinates to device
    int* d_pointsX;
    int* d_pointsY;
    CUDA_CALL(cudaMalloc((void**)&d_pointsX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_pointsY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_pointsX, h_pointsX, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pointsY, h_pointsY, N * sizeof(int), cudaMemcpyHostToDevice));

    // Result come to hull - max size is needed because all coordinate can be the part of the hull
    int h_hullX[N];
    int h_hullY[N];
    int h_hullSize = 0; // Counts how many points are added to the hull

    // Allocate hull on device
    int* d_hullX;
    int* d_hullY;
    int* d_hullSize;
    CUDA_CALL(cudaMalloc((void**)&d_hullX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_hullY, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_hullSize, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_hullSize, sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_hullSize, &h_hullSize, sizeof(int), cudaMemcpyHostToDevice));

    // Get min and max x-coordinate values from pointsX with parallel reduction
    int* d_min;
    int* d_max;
    CUDA_CALL(cudaMalloc((void**)&d_min, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_max, sizeof(int)));

    reduceMin<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_min, d_pointsX);
    //reduceMin<<<1, blockSize, blockSize>>>(d_minX, d_minX);
    reduceMax<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_max, d_pointsX);
    //reduceMax<<<1, blockSize, blockSize>>>(d_maxX, d_maxX);

    // Get the index of min and max values
    int* d_minIdx;
    int* d_maxIdx;
    CUDA_CALL(cudaMalloc((void**)&d_minIdx, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_maxIdx, sizeof(int)));

    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_minIdx, d_min, d_pointsX);
    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_maxIdx, d_max, d_pointsX);

    CUDA_CALL(cudaFree(d_min));
    CUDA_CALL(cudaFree(d_max));

    // Check recursive on both sides of the min and max defined line
    quickHull(d_hullX, d_hullY, d_hullSize, d_minIdx, d_maxIdx, d_pointsX, d_pointsY, 1);
    quickHull(d_hullY, d_hullY, d_hullSize, d_minIdx, d_maxIdx, d_pointsX, d_pointsY, -1);

    // Copy results aka hull to host for print
    CUDA_CALL(cudaMemcpy(h_hullX, d_hullX, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(h_hullY, d_hullY, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&h_hullSize, d_hullSize, sizeof(int), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < h_hullSize; i++)
        printf("(%d, %d), ", h_hullX[i], h_hullY[i]);
    printf("\n");

    // Clean up
    CUDA_CALL(cudaFree(d_minIdx));
    CUDA_CALL(cudaFree(d_maxIdx));
    CUDA_CALL(cudaFree(d_pointsX));
    CUDA_CALL(cudaFree(d_pointsY));
    CUDA_CALL(cudaFree(d_hullX));
    CUDA_CALL(cudaFree(d_hullY));
}
