#include <iostream>
#include <cuda_runtime.h>

#define N 3  // Rows of A and C
#define M 2  // Columns of A and Rows of B
#define P 4  // Columns of B and C

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index of C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index of C

    if (row < n && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

int main() {
    float h_A[N * M] = { 1, 2, 3, 4, 5, 6 };              // 3x2 matrix
    float h_B[M * P] = { 1, 2, 3, 4,                      // 2x4 matrix
                         5, 6, 7, 8 };
    float h_C[N * P] = {0};                              // 3x4 output

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, N * M * sizeof(float));
    cudaMalloc(&d_B, M * P * sizeof(float));
    cudaMalloc(&d_C, N * P * sizeof(float));

    cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + 15) / 16, (N + 15) / 16);
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N, M, P);

    cudaMemcpy(h_C, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Resultant Matrix C (3x4):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << h_C[i * P + j] << " ";
        }
        std::cout << "\n";
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
