// 

#include <curand.h>

void RandomGenerator()
{
    //size_t n = 10;
    //curandGenerator_t gen;
    //unsigned int* devData;
    //unsigned int* hostData;

    ///* Allocate n floats on host */
    //hostData = (unsigned*)calloc(n, sizeof(unsigned int));

    ///* Allocate n floats on device */
    //CUDA_CALL(cudaMalloc((void**)&devData, n * sizeof(unsigned int)));

    ///* Create pseudo-random number generator */
    //CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    ///* Set seed */ 
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    ///* Generate n floats on device */
    //CURAND_CALL(curandGenerate(gen, devData, n));

    ///* Copy device memory to host */
    //CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    ///* Show result */
    //for (size_t i = 0; i < n; i++) {
    //    printf("%d ", hostData[i]);
    //}
    //printf("\n");

    ///* Cleanup */
    //CURAND_CALL(curandDestroyGenerator(gen));
    //CUDA_CALL(cudaFree(devData));
    //free(hostData);
    //return EXIT_SUCCESS;

    //int threadX = 32;
    //int threadY = 32;
    //int threadCount = N;
    //int blockCount = (threadCount - 1) / N + 1;

    //generatePoints<<<blockCount, dim3(threadX, threadY)>>>();
    //cudaError_t err = cudaSuccess;
    //err = cudaGetLastError();
    //if (err != cudaSuccess)
       // printf("generatePoints error");
}
