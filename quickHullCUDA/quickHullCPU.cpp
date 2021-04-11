//

#include "quickHullCPU.h"

// Results - the hull - come here
int* hullX = (int*)malloc(256 * sizeof(int));
int* hullY = (int*)malloc(256 * sizeof(int));
size_t hullSize = 0;

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
            return -1;
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

        static auto findInHull = [](int x, int y, int* hullX, int* hullY, size_t hullSize)
        {
            for (size_t i = 0; i < hullSize; ++i)
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
            hullX[hullSize - 1] = x1;
            hullY[hullSize - 1] = y1;
        }
        if (!findInHull(x2, y2, hullX, hullY, hullSize))
        {
            ++hullSize;
            hullX[hullSize - 1] = x2;
            hullY[hullSize - 1] = y2;
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

    size_t minX = 0;
    size_t maxX = 0;
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
    for (size_t i = 0; i < hullSize; i++)
        printf("(%d, %d), ", hullX[i], hullY[i]);

    free(hullX);
    free(hullY);
}
