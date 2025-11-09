#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;

// size of the each vector 
int N =10000000;
int blockSize = 256;

// normal cpu execution code

void vector_addition_cpu( int* h_a, int* h_b, int* h_c, int size){
    for(int i =0;i<size;i++){
        h_c[i]=h_a[i]+h_b[i];
    }
}

// gpu kernel code
__global__ void(int*d_a , int*d_b, int*d_c,int size){
    int i =blockIdx.x*blockDim.x +threadIdx.x;
    if(i<size){
        d_c[i]=d_a[i]+d_b[i];
    }
}


// functioon to initialize the vectors
void vector_init(int* vect , int size){
    for( int i =0; i<size; i++){
        vect[i]=(float)(rand()/RAND_MAX);
    }
}

void get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e9;
}


int main(){
    int *h_a, *h_b, *h_output_cpu, *h_output_gpu;
    int *d_a, *d_b, *d_c;
    

    // memory allocation on host
    size_t bytes=N * sizeof(float);;
    h_a=(float*)malloc(bytes);
    h_b=(float*)malloc(bytes);
    h_output_cpu=(float*)malloc(bytes);
    h_output_gpu=(float*)malloc(bytes);

    // initialize the vectors
    srand(time(NULL));
    vector_init(h_a,N);
    vector_init(h_b,N);

    // memory allocation on device
    cudamalloc(&h_a, bytes);
    cudamalloc(&h_b, bytes);
    cudamalloc(&h_c, bytes);

    // copy data from host to device
    cudamemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudamemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // block size and grid size
    int num_blocks=(N+blockSize-1)/blockSize;

    printf("warming up \n");
    for (int i =0;i<5;i++){
        vector_addition_kernel<<<num_blocks, blockSize>>>(d_a, d_b, d_c, N);
        cudadevicesynchronize();
    }

    // Benchmarking
    print ("Starting CPU computation \n");
    double cpu_total_time=get_time();
    for (int i=0; i<100; i++){
        double iter_time_start=get_time();
        vector_addition_cpu(h_a, h_b, h_output_cpu, N);
        double iter_time_end=get_time();
        cpu_total_time += iter_time_end - iter_time_start;
    }
    double cpu_avg_time=cpu_total_time/100.0;
    printf("Average CPU time: %f ms \n", cpu_avg_time/1e6);


    print ("Starting GPU computation \n");
    double gpu_total_time=get_time();
    for (int i=0; i<100; i++){
        double iter_time_start=get_time();
        vector_addition_kernel<<<num_blocks, blockSize>>>(d_a, d_b, d_c, N);
        cudadevicesynchronize();
        double iter_time_end=get_time();
        gpu_total_time += iter_time_end - iter_time_start;
    }
    printf("Average GPU time: %f ms \n", gpu_avg_time/1e6);

    // verify the result
    cudamemcpy(h_output_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    bool match = true;
    for (int i -0; i< N; i++){
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5){
            match = false;
            printf("Results do not match! CPU: %f, GPU: %f \n", h_output_cpu[i], h_output_gpu[i]);
            break;
        }
    }
    printf("Results match: %d \n", match);

    // free memory
    free(h_a);
    free(h_b);
    free(h_output_cpu);
    free(h_output_gpu);
    cudafree(d_a);
    cudafree(d_b);
    cudafree(d_c);
    return 0;

}