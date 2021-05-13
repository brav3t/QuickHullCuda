//

#include "quickHullGPU.cuh"

// DEVICE
__device__ int d_hullX[N];
__device__ int d_hullY[N];
__device__ int d_hullSize;

// HOST
// Result come to hull - max size is needed because all coordinate can be the part of the hull.
int h_hullX[N];
int h_hullY[N];
int h_hullSize = 0; // Counts how many points are added to the hull.

void quickHullGPU(int* h_pointsX, int* h_pointsY)
{
    if (N < 3 && !h_pointsX && !h_pointsY)
        return;

    // Copy coordinates to device.
    int* d_pointsX;
    int* d_pointsY;
    CUDA_CALL(cudaMalloc((void**)&d_pointsX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_pointsY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_pointsX, h_pointsX, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pointsY, h_pointsY, N * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpyToSymbol(d_hullSize, &h_hullSize, sizeof(int)));

    // Get min and max x-coordinate values from pointsX with parallel reduction.
    int* d_min;
    int* d_max;
    CUDA_CALL(cudaMalloc((void**)&d_min, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_max, sizeof(int)));

    MinSearch<<<GRID_SIZE, BLOCK_SIZE>>>(d_min, d_pointsX);
    MaxSearch<<<GRID_SIZE, BLOCK_SIZE>>>(d_max, d_pointsX);

    // Get the index of min and max values.
    int* d_minIdx;
    int* d_maxIdx;
    CUDA_CALL(cudaMalloc((void**)&d_minIdx, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_maxIdx, sizeof(int)));
    getIndexOfValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_minIdx, d_min, d_pointsX);
    getIndexOfValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_maxIdx, d_max, d_pointsX);
    CUDA_CALL(cudaFree(d_min));
    CUDA_CALL(cudaFree(d_max));

    // Check recursive on both sides of the min and max defined line.
    int side = 1;
    int* d_side;
    CUDA_CALL(cudaMalloc((void**)&d_side, sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_side, &side, sizeof(int), cudaMemcpyHostToDevice));
    quickHull(d_minIdx, d_maxIdx, d_side, d_pointsX, d_pointsY);

    side = -1;
    CUDA_CALL(cudaMemcpy(d_side, &side, sizeof(int), cudaMemcpyHostToDevice));
    quickHull(d_minIdx, d_maxIdx, d_side, d_pointsX, d_pointsY);

    // Clean up.
    CUDA_CALL(cudaFree(d_minIdx));
    CUDA_CALL(cudaFree(d_maxIdx));
    CUDA_CALL(cudaFree(d_pointsX));
    CUDA_CALL(cudaFree(d_pointsY));

    // Copy results aka hull to host for print.
    CUDA_CALL(cudaMemcpyFromSymbol(h_hullX, d_hullX, N * sizeof(int)));
    CUDA_CALL(cudaMemcpyFromSymbol(h_hullY, d_hullY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpyFromSymbol(&h_hullSize, d_hullSize, sizeof(int)));

    for (size_t i = 0; i < h_hullSize; i++)
        printf("(%d, %d), ", h_hullX[i], h_hullY[i]);
    printf("\n");
}

void quickHull(int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int* d_expectedSide, int* d_pointsX, int* d_pointsY)
{
    // Calculate all points distances from the line.
    int* d_distances;
    CUDA_CALL(cudaMalloc((void**)&d_distances, N * sizeof(int)));
    getDistancesFromLine<<<GRID_SIZE, BLOCK_SIZE>>>(d_distances, d_P1Idx, d_P2Idx, d_expectedSide, d_pointsX, d_pointsY);

    // Get the max distance from distances.
    int* d_maxDist;
    CUDA_CALL(cudaMalloc((void**)&d_maxDist, sizeof(int)));
    MaxSearch<<<GRID_SIZE, BLOCK_SIZE>>>(d_maxDist, d_distances);

    // Need to copy back maxDistance for the CPU to decide if points can be added to hull and recursion reached base condition.
    int h_maxDist = -1;
    CUDA_CALL(cudaMemcpy(&h_maxDist, d_maxDist, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_maxDist)
    {
        // Need to copy back hullSize for thread count.
        int h_hullSize;
        CUDA_CALL(cudaMemcpyFromSymbol(&h_hullSize, d_hullSize, sizeof(int)));
        int blockSize = h_hullSize + 1;
        tryAddToHull<<<1, blockSize>>>(d_P1Idx, d_P2Idx, d_pointsX, d_pointsY);
        return;
    }

    int* d_maxDistIdx;
    CUDA_CALL(cudaMalloc((void**)&d_maxDistIdx, sizeof(int)));
    getIndexOfValue<<<GRID_SIZE, BLOCK_SIZE>>>(d_maxDistIdx, d_maxDist, d_distances); // No problem if there are more equal values.

    CUDA_CALL(cudaFree(d_maxDist));
    CUDA_CALL(cudaFree(d_distances));

    // Calculate left side of max distance.
    int* d_side;
    CUDA_CALL(cudaMalloc((void**)&d_side, sizeof(int)));
    getSide<<<1, 1>>>(d_side, d_maxDistIdx, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY); // This is just awfull.
    quickHull(d_maxDistIdx, d_P1Idx, d_side, d_pointsX, d_pointsY);

    // Calculate rigth side of max distance.
    getSide<<<1, 1>>>(d_side, d_maxDistIdx, d_P2Idx, d_P1Idx, d_pointsX, d_pointsY);
    quickHull(d_maxDistIdx, d_P2Idx, d_side, d_pointsX, d_pointsY);

    CUDA_CALL(cudaFree(d_side));
}

// DEVICE

__global__ static void MinSearch(int* min, int* array)
{
    __shared__ int localMin[BLOCK_SIZE * 2];
    int blockSize = BLOCK_SIZE;
    int itemc1 = threadIdx.x * 2;
    int itemc2 = threadIdx.x * 2 + 1;
    for (int k = 0; k <= 1; k++) {
        int blockStart = blockIdx.x * blockDim.x * 4 + k * blockDim.x * 2;
        int loadIndx = threadIdx.x + blockDim.x * k;
        if (blockStart + itemc2 < N) {
            int value1 = array[blockStart + itemc1];
            int value2 = array[blockStart + itemc2];
            localMin[loadIndx] = value1 < value2 ? value1 : value2;
        }
        else
            if (blockStart + itemc1 < N)
                localMin[loadIndx] = array[blockStart + itemc1];
            else
                localMin[loadIndx] = array[0];
    }
    __syncthreads();
    
    while (blockSize > 0)
    {
        int locMin = localMin[itemc1] < localMin[itemc2] ? localMin[itemc1] : localMin[itemc2];
        __syncthreads();
        localMin[threadIdx.x] = locMin;
        __syncthreads();
        blockSize = blockSize / 2;
    }
    if (threadIdx.x == 0)
    {
        *min = MAX_COORD;
        atomicMin(min, localMin[0]);
    }
}

__global__ static void MaxSearch(int* max, int* array)
{
    __shared__ int localMax[BLOCK_SIZE * 2];
    int blockSize = BLOCK_SIZE;
    int itemc1 = threadIdx.x * 2;
    int itemc2 = threadIdx.x * 2 + 1;
    for (int k = 0; k <= 1; k++) {
        int blockStart = blockIdx.x * blockDim.x * 4 + k * blockDim.x * 2;
        int loadIndx = threadIdx.x + blockDim.x * k;
        if (blockStart + itemc2 < N) {
            int value1 = array[blockStart + itemc1];
            int value2 = array[blockStart + itemc2];
            localMax[loadIndx] = value1 > value2 ? value1 : value2;
        }
        else
            if (blockStart + itemc1 < N)
                localMax[loadIndx] = array[blockStart + itemc1];
            else
                localMax[loadIndx] = array[0];
    }
    __syncthreads();

    while (blockSize > 0)
    {
        int locMax = localMax[itemc1] > localMax[itemc2] ? localMax[itemc1] : localMax[itemc2];
        __syncthreads();
        localMax[threadIdx.x] = locMax;
        __syncthreads();
        blockSize = blockSize / 2;
    }
    if (threadIdx.x == 0)
    {
        atomicMax(max, localMax[0]);
    }
}

__global__ void getIndexOfValue(int* index, int* value, int* array)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    if (gid == 0)
        *index = -1;

    __syncthreads(); // Causes warp but needed for index not get overwritten.

    if (array[gid] == *value)
    {
        *index = gid;
        //printf("index = %d\n", *index);
    }
}

__global__ void getDistancesFromLine(int* distances, int* P1Idx, int* P2Idx, int* expectedSide, int* pointsX, int* pointsY)
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

    if (side == *expectedSide)
    {
        // Distance of P0 from P1-P2 line
        int dist = abs((y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1));
        distances[gid] = dist;
    }
    else
        distances[gid] = 0;

    //printf("%d ", distances[gid]);
}

__global__ void getSide(int* side, int* maxDistIdx, int* P1Idx, int* P2Idx, int* pointsX, int* pointsY)
{
    int x1 = pointsX[*P1Idx];
    int y1 = pointsY[*P1Idx];
    int x2 = pointsX[*P2Idx];
    int y2 = pointsY[*P2Idx];
    int x0 = pointsX[*maxDistIdx];
    int y0 = pointsY[*maxDistIdx];

    int area = (y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1);
    if (0 < area)
        *side = 1;
    if (area < 0)
        *side = -1;
    *side = -(*side);

    //printf("%d ", *side);
};

__global__ void tryAddToHull(int* P1Idx, int* P2Idx, int* pointsX, int* pointsY)
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

    if (tid < d_hullSize)
    {
        if (d_hullX[gid] == pointsX[*P1Idx] && d_hullY[gid] == pointsY[*P1Idx])
            sh_foundP1 = true;

        if (d_hullX[gid] == pointsX[*P2Idx] && d_hullY[gid] == pointsY[*P2Idx])
            sh_foundP2 = true;
    }

    __syncthreads();

    if (!tid && !sh_foundP1)
    {
        d_hullX[d_hullSize] = pointsX[*P1Idx];
        d_hullY[d_hullSize] = pointsY[*P1Idx];
        ++(d_hullSize);
    }

    if (!tid && !sh_foundP2)
    {
        d_hullX[d_hullSize] = pointsX[*P2Idx];
        d_hullY[d_hullSize] = pointsY[*P2Idx];
        ++(d_hullSize);
    }
}
