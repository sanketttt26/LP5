#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()

using namespace std;

// CUDA kernel to perform vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N;
    cout << "Enter size of vectors: ";
    cin >> N;

    int size = N * sizeof(int);

    // Allocate host memory
    int *A = (int *)malloc(size);
    int *B = (int *)malloc(size);
    int *C = (int *)malloc(size);

    // Seed for random number generation
    srand(time(0));

    // Generate random values for A and B
    for (int i = 0; i < N; i++)
    {
        A[i] = rand() % 100; // random values from 0 to 99
        B[i] = rand() % 100;
    }

    // Print generated vectors
    cout << "\nGenerated Vector A:\n";
    for (int i = 0; i < N; i++)
    {
        cout << "A[" << i << "] = " << A[i] << endl;
    }

    cout << "\nGenerated Vector B:\n";
    for (int i = 0; i < N; i++)
    {
        cout << "B[" << i << "] = " << B[i] << endl;
    }

    // Allocate device memory
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy host data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print results
    cout << "\nResult of A + B:\n";
    for (int i = 0; i < N; i++)
    {
        cout << "A[" << i << "] + B[" << i << "] = " << C[i] << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
