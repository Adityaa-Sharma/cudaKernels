#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void whoami(void){
    int block_id = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

    int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;

    int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    int id = block_offset + thread_offset;

    printf("Hello from thread %d | Block ID: %d | Block Dim: (%d,%d,%d) | Thread ID: (%d,%d,%d)\n", 
           (int)id, (int)block_id, 
           (int)blockDim.x, (int)blockDim.y, (int)blockDim.z,
           (int)threadIdx.x, (int)threadIdx.y, (int)threadIdx.z);
}

int main() {
    dim3 grid(2, 2, 1);   
    dim3 block(2, 2, 1);  

    whoami<<<grid, block>>>(); // this isn the host function call
    cudaDeviceSynchronize();
    
    return 0;
}