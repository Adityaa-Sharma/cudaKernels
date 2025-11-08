#include <iostream>
using namespace std;

__global__void whoami(void){
    int block_id=blockIdx.x+blockIdx.y*gridDim.x +blockIdx.z*gridDim.x*gridDim.y;

    const int block_offset=block_id*blockDim.x*blockDim.y*blockDim.z;

    const int thread_offset=threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;

    const int id = block_offset + thread_offset;

    cout << "Hello from thread "<< id << " Block ID: "<< block_id << " Block Dim:" << (blockDim.x, blockDim.y, blockDim.z) << " Thread ID: " << (threadIdx.x, threadIdx.y, threadIdx.z) << endl;
}

int main() {
    dim3 grid(2,2,1);
    dim3 block(2,2,1);

    whoami<<<grid,block>>>();
    cudaDeviceSynchronize();
    return 0;
}