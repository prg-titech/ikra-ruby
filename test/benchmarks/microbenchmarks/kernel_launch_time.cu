#include <chrono>
#include <stdio.h>

#define GRID_DIM 8
#define BLOCK_DIM 256
#define ITERATIONS 1000000

using namespace std;

__global__ void kernel()
{

}

int main()
{
    // Measure kernel invocation
    chrono::high_resolution_clock::time_point start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < ITERATIONS; i++)
    {
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
    }

    chrono::high_resolution_clock::time_point end_time = chrono::high_resolution_clock::now();
    long kernel_time_elapsed = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    printf("Kernel invocation time: %i ms\n", kernel_time_elapsed / 1000);

    // Measure loop overhead
    start_time = chrono::high_resolution_clock::now();
    
    for (int i = 0; i < ITERATIONS; i++)
    {
        kernel<<<GRID_DIM, BLOCK_DIM>>>();
    }

    end_time = chrono::high_resolution_clock::now();
    long loop_time_elapsed = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    printf("Loop overhead: %i ms\n", loop_time_elapsed / 1000);   
}