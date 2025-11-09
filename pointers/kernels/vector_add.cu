#include <iostream>
#include <cuda_runtime.h>
#include <time.h>
using namespace std;

// size of the each vector 
int N =100000000;
int blockSize = 256;

// normal cpu execution code

void vector_addition_cpu( float* h_a, float* h_b, float* h_c, int size){
    for(int i =0;i<size;i++){
        h_c[i]=h_a[i]+h_b[i];
    }
}

// gpu kernel code
__global__ void vector_addition_kernel(float*d_a , float*d_b, float*d_c,int size){
    int i =blockIdx.x*blockDim.x +threadIdx.x;
    if(i<size){
        d_c[i]=d_a[i]+d_b[i];
    }
}


// functioon to initialize the vectors
void vector_init(float* vect , float size){
    for( int i =0; i<size; i++){
        vect[i]=(float)(rand()/RAND_MAX);
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


int main(){
    float *h_a, *h_b, *h_output_cpu, *h_output_gpu;
    float *d_a, *d_b, *d_c;
    

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
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // block size and grid size
    int num_blocks=(N+blockSize-1)/blockSize;

    printf("warming up \n");
    for (int i =0;i<5;i++){
        vector_addition_kernel<<<num_blocks, blockSize>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmarking
    printf("Starting CPU computation \n");
    double cpu_total_time=0.0;
    for (int i=0; i<1000; i++){
        double iter_time_start=get_time();
        vector_addition_cpu(h_a, h_b, h_output_cpu, N);
        double iter_time_end=get_time();
        cpu_total_time += iter_time_end - iter_time_start;
    }
    double cpu_avg_time=cpu_total_time/1000.0;
    printf("Average CPU time: %f s \n", cpu_avg_time);


    printf("Starting GPU computation \n");
    double gpu_total_time=0.0;
    for (int i=0; i<100; i++){
        double iter_time_start=get_time();
        vector_addition_kernel<<<num_blocks, blockSize>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double iter_time_end=get_time();
        gpu_total_time += iter_time_end - iter_time_start;
    }
    double gpu_avg_time=gpu_total_time/100.0;
    printf("Average GPU time: %f s \n", gpu_avg_time);

    // verify the result
    cudaMemcpy(h_output_gpu, d_c, bytes, cudaMemcpyDeviceToHost);
    bool match = true;
    for (int i=0; i< N; i++){
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
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;

}