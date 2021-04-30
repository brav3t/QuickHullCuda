//

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define N 10
#define BLOCK_SIZE 1024

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("\nError at %s line:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE); } } while(0)

//#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
//    printf("Error at %s line:%d\n",__FILE__,__LINE__);\
//    return EXIT_FAILURE;}} while(0)
