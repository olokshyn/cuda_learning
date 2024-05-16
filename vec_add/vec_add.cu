#include <cstdio>

#define CUDA_CHECK(call)                                                                  \
    {                                                                                     \
        cudaError_t err = (call);                                                         \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

__global__ void vec_add_kernel(const float *A, const float *B, float *C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vec_add(const float *A_h, const float *B_h, float *C_h, int n)
{
    // Allocate memory on the device
    float *A_d;
    CUDA_CHECK(cudaMalloc(&A_d, n * sizeof(float)));
    float *B_d;
    CUDA_CHECK(cudaMalloc(&B_d, n * sizeof(float)));
    float *C_d;
    CUDA_CHECK(cudaMalloc(&C_d, n * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(A_d, A_h, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch the kernel
    vec_add_kernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Copy data from device to host
    CUDA_CHECK(cudaMemcpy(C_h, C_d, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free memory on the device
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

int main()
{
    const int n = 1 << 20;

    // Allocate memory on the host
    float *A = (float *)malloc(n * sizeof(float));
    float *B = (float *)malloc(n * sizeof(float));
    float *C = (float *)malloc(n * sizeof(float));

    for (int i = 0; i < n; ++i)
    {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    vec_add(A, B, C, n);

    free(A);
    free(B);
    free(C);
}
