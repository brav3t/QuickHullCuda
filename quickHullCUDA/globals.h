//

#include <stdio.h>
#include <stdlib.h>

#define N10 10
#define N1000 1000

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("\nError at %s line:%d\n",__FILE__,__LINE__); exit(EXIT_FAILURE); } } while(0)

//#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
//    printf("Error at %s line:%d\n",__FILE__,__LINE__);\
//    return EXIT_FAILURE;}} while(0)
