#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMul(float *A, float *B, float *C, int N)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < N)
    {
        float value = 0;
        for (int k = 0; k < N; k++)
        {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main()
{
    int N;
    cout << "Enter the size N of the NxN matrices: ";
    cin >> N;

    size_t size = N * N * sizeof(float);
    float *A, *B, *C, *d_A, *d_B, *d_C;

    // Allocate memory on host
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Allocate memory on device
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Input matrices from user
    cout << "Enter elements for Matrix A (" << N * N << " values):" << endl;
    for (int i = 0; i < N * N; i++)
    {
        cin >> A[i];
    }

    cout << "Enter elements for Matrix B (" << N * N << " values):" << endl;
    for (int i = 0; i < N * N; i++)
    {
        cin >> B[i];
    }

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Display results
    cout << "\nMatrix A:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << A[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << B[i * N + j] << " ";
        }
        cout << endl;
    }

    cout << "\nMatrix C (Result of A x B):\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            cout << C[i * N + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
