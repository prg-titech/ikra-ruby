#include <chrono>
#include <stdio.h>

#define GRID_DIM 39063
#define BLOCK_DIM 256

using namespace std;

__global__ void kernel_new(int *data) {
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 10000000) return;

    int idx_0 =_tid_ / 500000;
    int idx_1 = (_tid_ / 1000) % 500;
    int idx_2 = (_tid_ / 2) % 500;
    int idx_3 = (_tid_ / 1) % 2;

    // int indices[] = {idx_0, idx_1, idx_2, idx_3};

    data[_tid_] = idx_2 % 133777;
}

__global__ void kernel_5(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 10000000) return;

    int idx_2 = (_tid_ / 2) % 500;

    new_data[_tid_] = (data[_tid_] + idx_2) % 13377;
}

int main()
{
    auto start_entire = chrono::high_resolution_clock::now();

    // Init
    cudaThreadSynchronize();

    long time_kernel = 0;
    long time_alloc = 0;
    long time_free = 0;
    long time_transfer = 0;

    // Measure kernel invocation
    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();
    long loop_time_elapsed;

    printf("START\n");
    int * data;
    cudaMalloc(&data, (sizeof(int) * 10000000));
    cudaThreadSynchronize();

    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

 
    start_time = chrono::high_resolution_clock::now();
    kernel_new<<<GRID_DIM, BLOCK_DIM>>>(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    for (int r = 0; r < 500; r++)
    {
        start_time = chrono::high_resolution_clock::now();
        int * new_data;
        cudaMalloc(&new_data, (sizeof(int) * 10000000));
        cudaThreadSynchronize();
        end_time = chrono::high_resolution_clock::now();
        time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        start_time = chrono::high_resolution_clock::now();
        kernel_5<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
        cudaThreadSynchronize();
        end_time = chrono::high_resolution_clock::now();
        time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        start_time = chrono::high_resolution_clock::now();
        cudaFree(data);
        cudaThreadSynchronize();
        end_time = chrono::high_resolution_clock::now();
        time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        data = new_data;
    }

    cudaThreadSynchronize();

    // Copy back
    start_time = chrono::high_resolution_clock::now();
    int * tmp_result = (int *) malloc(sizeof(int) * 10000000);
    cudaMemcpy(tmp_result, data, sizeof(int) * 10000000, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_transfer += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();


    end_time = chrono::high_resolution_clock::now();
    int time_entire = chrono::duration_cast<chrono::microseconds>(end_time - start_entire).count();

    printf("alloc: %f\n", time_alloc / 1000.0);
    printf("kernel: %f\n", time_kernel / 1000.0);
    printf("transfer: %f\n", time_transfer / 1000.0f);
    printf("free: %f\n", time_free / 1000.f);
    printf("rest: %f\n", (time_entire - time_alloc - time_kernel - time_transfer - time_free) / 1000.0f);

    printf("END\n");  
}
