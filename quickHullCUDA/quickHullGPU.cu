//

#include "quickHullGPU.cuh"

// shitty function should be replaced with parallel reduction for min and max values
__global__ void getMinMax(int* minX, int* maxX, int* pointsX, int N)
{
    int globalX = blockIdx.x * blockDim.x + threadIdx.x;

    if (N < globalX)
        return;

    *minX = 0;
    *maxX = 0;

    // this can give false results because of racing conditions
    if (pointsX[globalX] < pointsX[*minX])
    {
        atomicExch(minX, globalX);
        printf("minX = %d\n", *minX);
    }

    // this can give false results because of racing conditions
    if (pointsX[*maxX] < pointsX[globalX])
    {
        atomicExch(maxX, globalX);
        printf("maxX = %d\n", *maxX);
    }
}

__global__ void quickHull(int* hullX, int* hullY, int hullSize, int x1, int y1, int x2, int y2, int side)
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
            return -1;
        return 0;
    };

    int idxMaxDist = -1;
    int maxDist = 0;

    // Get the point with max distance from line.
    //for (size_t i = 0; i < _N; ++i)
    //{
    //    int dist = getDistFromLine(x1, y1, x2, y2, _pointsX[i], _pointsY[i]);
    //    if (getSide(x1, y1, x2, y2, _pointsX[i], _pointsY[i]) == side && maxDist < dist)
    //    {
    //        idxMaxDist = i;
    //        maxDist = dist;
    //    }
    //}

    // If no point found then add the end points P1 and P2 of line to hull.
    //if (idxMaxDist == -1)
    //{
    //    // Resize hull containter if not enought space for two new points.
    //    static int hullArraySize = 256;
    //    if (hullArraySize < _hullSize + 2)
    //    {
    //        hullArraySize *= 2;
    //        if (void* new_hullX = realloc(_hullX, hullArraySize)) // error handling
    //            _hullX = (int*)new_hullX;
    //        if (void* new_hullY = realloc(_hullY, hullArraySize))
    //            _hullY = (int*)new_hullY;
    //    }

    //    static auto findInHull = [](int x, int y, int* _hullX, int* _hullY, size_t _hullSize)
    //    {
    //        for (size_t i = 0; i < _hullSize; ++i)
    //        {
    //            if (_hullX[i] == x && _hullY[i] == y)
    //                return true;
    //        }
    //        return false;
    //    };

    //    // Add end of lines to hull. Check for duplicats.
    //    if (!findInHull(x1, y1, _hullX, _hullY, _hullSize))
    //    {
    //        ++_hullSize;
    //        _hullX[_hullSize - 1] = x1;
    //        _hullY[_hullSize - 1] = y1;
    //    }
    //    if (!findInHull(x2, y2, _hullX, _hullY, _hullSize))
    //    {
    //        ++_hullSize;
    //        _hullX[_hullSize - 1] = x2;
    //        _hullY[_hullSize - 1] = y2;
    //    }

    //    return;
    //}

    //// Check for sides divided by idxMaxDist.
    //quickHull(_pointsX[idxMaxDist], _pointsY[idxMaxDist], x1, y1, -getSide(_pointsX[idxMaxDist], _pointsY[idxMaxDist], x1, y1, x2, y2));
    //quickHull(_pointsX[idxMaxDist], _pointsY[idxMaxDist], x2, y2, -getSide(_pointsX[idxMaxDist], _pointsY[idxMaxDist], x2, y2, x1, y1));
}

void quickHullGPU(int* h_pointsX, int* h_pointsY, size_t N)
{
    if (N < 3)
        return;

    int deviceNumber = 0;
    cudaSetDevice(deviceNumber);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceNumber);

    // The problem space to GPU
    int* d_pointsX;
    int* d_pointsY;
    
    // Copy problem space to GPU
    CUDA_CALL(cudaMalloc((void**)&d_pointsX, N * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_pointsY, N * sizeof(int)));
    CUDA_CALL(cudaMemcpy(d_pointsX, h_pointsX, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_pointsY, h_pointsY, N * sizeof(int), cudaMemcpyHostToDevice));

    // Index of the points with min and max value from points x coordinate.
    int* d_minX;
    int* d_maxX;
    CUDA_CALL(cudaMalloc((void**)&d_minX, sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_maxX, sizeof(int)));

    // calculate block and threads here
    
    // Get minX and maxX values from pointsX
    getMinMax<<<1, N>>>(d_minX, d_maxX, d_pointsX, N);

    // Results - the hull - came here
    int* d_hullX;
    int* d_hullY;
    size_t hullSize = 256;
    CUDA_CALL(cudaMalloc((void**)&d_hullX, hullSize * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_hullY, hullSize * sizeof(int)));

    // calculate block and threads here

    //quickHull<<<1, N>>>(d_hullX, d_hullY, hullSize, d_pointsX[d_minX], d_pointsY[d_minX], d_pointsX[d_maxX], d_pointsY[d_maxX], 1);
    //quickHull<<<1, N>>>(d_hullX, d_hullY, hullSize, d_pointsX[d_minX], d_pointsY[d_minX], d_pointsX[d_maxX], d_pointsY[d_maxX], -1);

    // clean up
    cudaError_t err2 = cudaFree(d_pointsX);
    //CUDA_CALL(cudaFree(d_pointsX));
    CUDA_CALL(cudaFree(d_pointsY));
    CUDA_CALL(cudaFree(d_minX));
    CUDA_CALL(cudaFree(d_maxX));
    CUDA_CALL(cudaFree(d_hullX));
    CUDA_CALL(cudaFree(d_hullY));
}
