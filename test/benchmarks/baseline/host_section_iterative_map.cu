#include <chrono>
#include <stdio.h>

#define NUM_THREADS 511
#define ITERATIONS 100000

using namespace std;

__global__ void kernel_init(int *values)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < NUM_THREADS)
    {
        values[tid] = tid;
    }
}

__global__ void kernel_map(int *values, int *next_values)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < NUM_THREADS)
    {
        next_values[tid] = values[tid] + 1;
    }
}

int main()
{
    chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    // Calculate dimensions
    int grid_dim = max((int) ceil(((float) NUM_THREADS) / 256), 1);
    int block_dim = (NUM_THREADS >= 256 ? 256 : NUM_THREADS);

    int *device_values;
    cudaMalloc(&device_values, (sizeof(int) * NUM_THREADS));

    kernel_init<<<grid_dim, block_dim>>>(device_values);
    for (int i = 0; i < ITERATIONS; i++)
    {
        int *next_device_values;
        cudaMalloc(&next_device_values, (sizeof(int) * NUM_THREADS));
        kernel_map<<<grid_dim, block_dim>>>(device_values, next_device_values);

        device_values = next_device_values;
    }

    // TODO: Copy data back to the host
    int *host_values = (int *) malloc(NUM_THREADS * sizeof(int));
    cudaMemcpy(host_values, device_values, NUM_THREADS * sizeof(int), cudaMemcpyDeviceToHost);

    chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
    long time_elapsed = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
    printf("Done. Took %i ms\n", time_elapsed / 1000);
}