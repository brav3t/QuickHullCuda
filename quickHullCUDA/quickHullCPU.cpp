//

#include "quickHullCPU.h"

// Results come here.
int _hullX[N];
int _hullY[N];
int _hullSize = 0;

int* _pointsX;
int* _pointsY;

// Main function that is called recursively.
void quickHull(int P1Idx_, int P2Idx_, int expectedSide_);

// 
void quickHullCPU(int* pointsX_, int* pointsY_)
{
    if (N < 3 && !pointsX_ && !pointsY_)
        return;

    _pointsX = pointsX_;
    _pointsY = pointsY_;

    int minIdx = 0;
    int maxIdx = 0;
    for (int i = 1; i < N; ++i)
    {
        if (_pointsX[i] < _pointsX[minIdx])
            minIdx = i;

        if (_pointsX[maxIdx] < _pointsX[i])
            maxIdx = i;
    }

    // Run for both sides of the minIdx and maxIdx defied line.
    quickHull(minIdx, maxIdx, 1);
    quickHull(minIdx, maxIdx, -1);

    // Print out results for debug.
    for (size_t i = 0; i < _hullSize; ++i)
        printf("(%d, %d), ", _hullX[i], _hullY[i]);
    printf("\n");
}

void quickHull(int P1Idx_, int P2Idx_, int expectedSide_)
{
    // P1
    int x1 = _pointsX[P1Idx_];
    int y1 = _pointsY[P1Idx_];
    // P2
    int x2 = _pointsX[P2Idx_];
    int y2 = _pointsY[P2Idx_];

    // Get value proportional to distance from line P1(x1, y1) P2(x2, y2) to P0(x0, y0).
    static auto getDistFromLine = [](int x1, int y1, int x2, int y2, int x0, int y0)
    {
        return abs((y0 - y1) * (x2 - x1) - (y2 - y1) * (x0 - x1));
    };

    // Gets side of point P0(x0, y0) from line defined by P1(x1, y1) and P2(x2, y2).
    static auto getSide = [](int x1, int y1, int x2, int y2, int x0, int y0)
    {
        int area = (y0 - y1) * (x2 - x1) - (x0 - x1) * (y2 - y1);
        if (0 < area)
            return 1;
        if (area < 0)
            return -1;
        return 0;
    };

    int maxDist = 0;
    int maxDistIdx = -1;

    // Get the point with max distance from line.
    for (int i = 0; i < N; ++i)
    {
        int dist = getDistFromLine(x1, y1, x2, y2, _pointsX[i], _pointsY[i]);
        if (getSide(x1, y1, x2, y2, _pointsX[i], _pointsY[i]) == expectedSide_ && maxDist < dist)
        {
            maxDist = dist;
            maxDistIdx = i;
        }
    }

    // If no point found then add the end points P1 and P2 of line to hull.
    if (maxDistIdx == -1)
    {
        static auto findInHull = [](int x, int y)
        {
            for (int i = 0; i < _hullSize; ++i)
            {
                if (_hullX[i] == x && _hullY[i] == y)
                    return true;
            }
            return false;
        };

        // Add end of lines to hull. Check for duplicats.
        if (!findInHull(x1, y1))
        {
            _hullX[_hullSize] = x1;
            _hullY[_hullSize] = y1;
            ++_hullSize;
        }
        if (!findInHull(x2, y2))
        {
            _hullX[_hullSize] = x2;
            _hullY[_hullSize] = y2;
            ++_hullSize;
        }

        return;
    }

    // Check for sides divided by maxDistIdx.
    quickHull(maxDistIdx, P1Idx_, -getSide(_pointsX[maxDistIdx], _pointsY[maxDistIdx], x1, y1, x2, y2));
    quickHull(maxDistIdx, P2Idx_, -getSide(_pointsX[maxDistIdx], _pointsY[maxDistIdx], x2, y2, x1, y1));
}
