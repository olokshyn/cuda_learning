#include <stdio.h>

#define CUDA_CHECK(call)                                                                  \
    {                                                                                     \
        cudaError_t err = (call);                                                         \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }
