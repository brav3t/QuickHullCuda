//

#include "quickHullGPU.cuh"

int h_cHullX[N];
int h_cHullY[N];
unsigned int h_cHullSize = 0;

int* h_pointsX;
int* h_pointsY;

const unsigned int gridSize = (N - 1) / BLOCK_SIZE + 1;

// Has warp
// Has shared memory bank conflicts
// Has instruction overhead
__global__ void reduceMin(int* g_min, int* g_points)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    extern __shared__ int s_min[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    s_min[tid] = g_points[gid];

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMin(&s_min[i], s_min[i + s]);

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_min[blockIdx.x] = s_min[0];
        printf("min = %d\n", s_min[tid]);
    }
}

__global__ void reduceMax(int* g_max, int* g_points)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (N <= gid)
        return;

    extern __shared__ int s_max[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;

    s_max[tid] = g_points[gid];

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2)
    {
        int i = 2 * s * tid;
        if (i < blockDim.x && (i + s) < N)
            atomicMax(&s_max[i], s_max[i + s]);

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
    {
        g_max[blockIdx.x] = s_max[0];
        printf("max = %d\n", s_max[tid]);
    }
}

__global__ void getIndexOfValue(int* g_index, int* g_value, int* g_array)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    if (gid == 0)
        *g_index = -1;

    __syncthreads(); //causes warp but needed for index not get overwritten

    if (g_array[gid] == *g_value)
    {
        *g_index = gid;
        printf("index = %d\n", *g_index);
    }
}

__global__ void getDistancesFromLine(int* g_distances, int* g_P1Idx, int* g_P2Idx, int* g_pointsX, int* g_pointsY, int side_)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (N <= gid)
        return;

    // P1
    int x1 = g_pointsX[*g_P1Idx];
    int y1 = g_pointsY[*g_P1Idx];
    // P2
    int x2 = g_pointsX[*g_P2Idx];
    int y2 = g_pointsY[*g_P2Idx];

    // P0
    int x0 = g_pointsX[gid];
    int y0 = g_pointsY[gid];

    // Get side of P0 from P1-P2 line
    int side = 0;
    int area = (y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1);
    if (0 < area)
        side = 1;
    else if (area < 0)
        side = -1;

    if (side == side_)
    {
        // Distance of P0 from P1-P2 line
        int dist = abs((y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1));
        g_distances[gid] = dist;
    }
    else
        g_distances[gid] = 0;

    printf("distances[%d] = %d\n", gid, g_distances[gid]);
}

void findInHull(int* found, int* point)
{

}

void quickHull(int* d_P1Idx /*Left point of line*/, int* d_P2Idx /*Right point of line*/, int* d_pointsX, int* d_pointsY, int side)
{
    int* d_distances;
    CUDA_CALL(cudaMalloc((void**)&d_distances, N * sizeof(int)));
    getDistancesFromLine<<<gridSize, BLOCK_SIZE>>>(d_distances, d_P1Idx, d_P2Idx, d_pointsX, d_pointsY, side);

    int* d_maxDist;
    CUDA_CALL(cudaMalloc((void**)&d_maxDist, gridSize * sizeof(int)));
    reduceMax<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_maxDist, d_distances);

    // Ezt egy add to cHull fv-be?
    int h_maxDist = 0;
    CUDA_CALL(cudaMemcpy(&h_maxDist, d_maxDist, sizeof(int), cudaMemcpyDeviceToHost));
    if (!h_maxDist)
    {
        int P1Idx;
        int P2Idx;
        CUDA_CALL(cudaMemcpy(&P1Idx, d_P1Idx, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(&P2Idx, d_P2Idx, sizeof(int), cudaMemcpyDeviceToHost));

        //findInHull(P1Idx);
        h_cHullX[h_cHullSize] = h_pointsX[P1Idx];
        h_cHullY[h_cHullSize] = h_pointsY[P1Idx];
        ++h_cHullSize;

        //findInHull(P12dx);
        h_cHullX[h_cHullSize] = h_pointsX[P2Idx];
        h_cHullY[h_cHullSize] = h_pointsY[P2Idx];
        ++h_cHullSize;

        return;
    } 
    /////////////////////

    int* d_maxDistIdx;
    CUDA_CALL(cudaMalloc((void**)&d_maxDistIdx, sizeof(int)));
    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_maxDistIdx, d_maxDist, d_distances); // ha több max érték van bármelyik beleírhatódik, nem baj a versenyhelyzet

    CUDA_CALL(cudaFree(d_distances));

    quickHull(d_P1Idx, d_maxDistIdx, d_pointsX, d_pointsY, side);
    quickHull(d_maxDistIdx, d_P2Idx, d_pointsX, d_pointsY, side);
}

void quickHullGPU(int* h_pointsX_, int* h_pointsY_)
{
    if (N < 3 && !h_pointsX_ && !h_pointsY_)
        return;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    h_pointsX = h_pointsX_;
    h_pointsY = h_pointsY_;

    // Copy coordinates to device
    int* d_pointsX;
    int* d_pointsY;
    CUDA_CALL(cudaMalloc((void**)&d_pointsX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_pointsY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_pointsX, h_pointsX, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pointsY, h_pointsY, N * sizeof(int), cudaMemcpyHostToDevice));

    // Get min and max x-coordinate values from pointsX with parallel reduction
    int* d_minX;
    int* d_maxX;
    CUDA_CALL(cudaMalloc((void**)&d_minX, gridSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_maxX, gridSize * sizeof(int)));

    reduceMin<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_minX, d_pointsX);
    //reduceMin<<<1, blockSize, blockSize>>>(d_minX, d_minX);
    reduceMax<<<gridSize, BLOCK_SIZE, BLOCK_SIZE>>>(d_maxX, d_pointsX);
    //reduceMax<<<1, blockSize, blockSize>>>(d_maxX, d_maxX);

    // Get the index of min and max values
    int* d_minIdx;
    int* d_maxIdx;
    CUDA_CALL(cudaMalloc((void**)&d_minIdx, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_maxIdx, sizeof(int)));

    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_minIdx, d_minX, d_pointsX);
    getIndexOfValue<<<gridSize, BLOCK_SIZE>>>(d_maxIdx, d_maxX, d_pointsX);

    CUDA_CALL(cudaFree(d_minX));
    CUDA_CALL(cudaFree(d_maxX));

    quickHull(d_minIdx, d_maxIdx, d_pointsX, d_pointsY, 1);
    quickHull(d_minIdx, d_maxIdx, d_pointsX, d_pointsY, -1);

    // Clean up
    CUDA_CALL(cudaFree(d_minIdx));
    CUDA_CALL(cudaFree(d_maxIdx));
    CUDA_CALL(cudaFree(d_pointsX));
    CUDA_CALL(cudaFree(d_pointsY));

    for (size_t i = 0; i < h_cHullSize; i++)
        printf("(%d, %d), ", h_cHullX[i], h_cHullY[i]);
    printf("\n");
}
