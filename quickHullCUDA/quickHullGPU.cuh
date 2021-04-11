//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "globals.h"

void quickHullGPU(int* h_pointsX, int* h_pointsY, size_t N);
