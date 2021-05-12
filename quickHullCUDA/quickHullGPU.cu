//

#include "quickHullGPU.cuh"

// DEVICE
__device__ int d_hullX[N];
__device__ int d_hullY[N];
__device__ int d_hullSize;

// Gets the minimum value from the array.
__global__ void reduceMin(int* min, int* array);
__global__ void reduceMax(int* max, int* array);
__global__ void getIndexOfValue(int* index, int* value, int* array);
// Returns and calculates all points distances from P1-P2 line. If side is not correct then distance is zero.
__global__ void getDistancesFromLine(int* distances, int* P1Idx, int* P2Idx, int* pointsX, int* pointsY, int expectedSide);
// Get the side of the point which has the max distance from P1-P2 line.
__global__ void getSide(int* side, int* maxDistIdx, int* P1Idx, int* P2Idx, int* pointsX, int* pointsY);
// If P1 or P2 is not in hull then add to it.
__global__ void tryAddToHull(int* P1Idx, int* P2Idx, int* pointsX, int* pointsY);

// HOST
// Result come to hull - max size is needed because all coordinate can be the part of the hull.
int h_hullX[N];
int h_hullY[N];
int h_hullSize = 0; // Counts how many points are added to the hull.

// Main function that is called recursively (On CPU).
void quickHull(int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int expectedSide, int* d_pointsX, int* d_pointsY);

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
    //CUDA_CALL(cudaMalloc((void**)&d_min, (N / BLOCK_SIZE) * sizeof(int)));
    //CUDA_CALL(cudaMalloc((void**)&d_max, (N / BLOCK_SIZE) * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_min, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_max, N * sizeof(int)));

    cudaDeviceProp deviceProp;
    CUDA_CALL(cudaGetDeviceProperties(&deviceProp, 0));
    int sharedMemPerBlock = deviceProp.sharedMemPerBlock;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int sh_count = sharedMemPerBlock / BLOCK_SIZE * sizeof(int);

    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reduceMin, 0, 0);
    gridSize = (N + blockSize - 1) / blockSize;
    
    unsigned int block_size = blockSize;
    unsigned int grid_size = ceil((N * N) / (block_size * 1.0));
    //reduceMin<<<grid_size, block_size, block_size>>>(d_min, d_pointsX);
    //reduceMax<<<grid_size, block_size, block_size>>>(d_max, d_pointsX);

    reduceMin<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE>>>(d_min, d_pointsX);
    //reduceMin<<<1, BLOCK_SIZE, BLOCK_SIZE>>>(d_min, d_min);
    reduceMax<<<GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE>>>(d_max, d_pointsX);
    //reduceMax<<<1, BLOCK_SIZE, BLOCK_SIZE>>>(d_max, d_max);

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
    quickHull(d_minIdx, d_maxIdx, 1, d_pointsX, d_pointsY);
    quickHull(d_minIdx, d_maxIdx, -1, d_pointsX, d_pointsY);

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

void quickHull(int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int expectedSide, int* d_pointsX, int* d_pointsY)
{
    // Calculate all points distances from the line.
    int* d_distances;
    CUDA_CALL(cudaMalloc((void**)&d_distances, N * sizeof(int)));
    getDistancesFromLine << <GRID_SIZE, BLOCK_SIZE >> > (d_distances, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY, expectedSide);

    // Get the max distance from distances.
    int* d_maxDist;
    CUDA_CALL(cudaMalloc((void**)&d_maxDist, sizeof(int)));
    reduceMax << <GRID_SIZE, BLOCK_SIZE, BLOCK_SIZE >> > (d_maxDist, d_distances);

    // Need to copy back max distance from line for CPU to decide that the recursion can be return, and the points can be added to hull.
    int h_maxDist = -1;
    CUDA_CALL(cudaMemcpy(&h_maxDist, d_maxDist, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_maxDist)
    {
        // Need to copy back hullSize for thread count.
        int h_hullSize;
        CUDA_CALL(cudaMemcpyFromSymbol(&h_hullSize, d_hullSize, sizeof(int)));
        int blockSize = h_hullSize + 1;
        tryAddToHull << <GRID_SIZE, blockSize >> > (d_P1Idx, d_P2Idx, d_pointsX, d_pointsY);
        return;
    }

    int* d_maxDistIdx;
    CUDA_CALL(cudaMalloc((void**)&d_maxDistIdx, sizeof(int)));
    getIndexOfValue << <GRID_SIZE, BLOCK_SIZE >> > (d_maxDistIdx, d_maxDist, d_distances); // No problem if there are more values.

    CUDA_CALL(cudaFree(d_distances));

    // Calculate left side of max distance.
    int* d_side;
    CUDA_CALL(cudaMalloc((void**)&d_side, sizeof(int)));
    getSide << <1, 1 >> > (d_side, d_maxDistIdx, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY); // This is just awfull.
    int h_side = 0;
    CUDA_CALL(cudaMemcpy(&h_side, d_side, sizeof(int), cudaMemcpyDeviceToHost));

    quickHull(d_maxDistIdx, d_P1Idx, -h_side, d_pointsX, d_pointsY);

    // Calculate rigth side of max distance.
    getSide << <1, 1 >> > (d_side, d_maxDistIdx, d_P2Idx, d_P1Idx, d_pointsX, d_pointsY);
    h_side = 0;
    CUDA_CALL(cudaMemcpy(&h_side, d_side, sizeof(int), cudaMemcpyDeviceToHost));

    quickHull(d_maxDistIdx, d_P2Idx, -h_side, d_pointsX, d_pointsY);
}

// DEVICE

// Has warp
// Has shared memory bank conflicts
// Has instruction overhead
__global__ void reduceMin(int* min, int* array)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    extern __shared__ int sh_min[];
    // Each thread loads one element from global to shared mem.
    unsigned int tid = threadIdx.x;
    int x = array[gid];
    sh_min[tid] = x;

    __syncthreads();

    // Do reduction in shared mem.
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMin(&sh_min[i], sh_min[i + s]);

        __syncthreads();
    }

    // Write result for this block to global mem.
    if (tid == 0)
    {
        min[blockIdx.x] = sh_min[0];
        //printf("min = %d\n", sh_min[tid]);
    }
}

// Has warp
// Has shared memory bank conflicts
// Has instruction overhead
__global__ void reduceMax(int* max, int* array)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    extern __shared__ int sh_max[];
    // Each thread loads one element from global to shared mem.
    unsigned int tid = threadIdx.x;

    sh_max[tid] = array[gid];

    __syncthreads();

    // Do reduction in shared mem.
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMax(&sh_max[i], sh_max[i + s]);

        __syncthreads();
    }

    // Write result for this block to global mem.
    if (tid == 0)
    {
        max[blockIdx.x] = sh_max[0];
        //printf("max = %d\n", sh_max[tid]);
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
        //printf("index = %d\n", *index);
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

    //printf("distances[%d] = %d\n", gid, distances[gid]);
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
}

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
