#include <chrono>
#include <stdio.h>

#define GRID_DIM 58594
#define BLOCK_DIM 1024

using namespace std;

// NEW:
// (indices[0] + indices[1]) % (indices[3] + indices[indices[1] % 4] + 7)

// STENCIL:
// pstencil([[-1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [-1, -1, 0, 0]], 37, with_index: true) do |values, indices|
//                 ((values[-1][0][0][0] % 938) + values[0][0][0][0] / 97) % 97717 + (indices[indices[indices[values[1][0][0][0] % 4] % 4] % 4] * (values[-1][-1][0][0] % 7) % 99)
//             end


// DIMENSIONS: [20, 500, 500, 12]

__global__ void kernel_new(int *data) {
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    data[_tid_] = (idx_0 + idx_1) % (idx_3 + indices[idx_1 % 4] + 7);
}

__global__ void kernel_1(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 + 1 >= 20) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_2(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_3(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_4(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_5(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_6(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_7(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_8(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_9(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

__global__ void kernel_10(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    int vi1 = (idx_0 - 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi2 = idx_0 * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi3 = (idx_0 + 1) * 500 * 500 * 12 + idx_1 * 500 * 12 + idx_2 * 12 + idx_3;
    int vi4 = (idx_0 - 1) * 500 * 500 * 12 + (idx_1 - 1) * 500 * 12 + idx_2 * 12 + idx_3;

    if (idx_0 - 1 < 0 || idx_1 - 1 < 0 || idx_0 >= 12) {
        new_data[_tid_] = 37;
        return;
    }

    int indices[] = {idx_0, idx_1, idx_2, idx_3};

    new_data[_tid_] = ((data[vi1] % 938) + data[vi2] / 97) % 97717 + (indices[indices[indices[data[vi3] % 4] % 4] % 4] * (data[vi4] % 7) % 99);
}

int main()
{
    long time_kernel = 0;
    long time_alloc = 0;
    long time_free = 0;
    long time_transfer = 0;
    long time_setup = 0;

    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();

    // Init
    start_time = chrono::high_resolution_clock::now();
    cudaThreadSynchronize();
    cudaFree(0);
    end_time = chrono::high_resolution_clock::now();;
    time_setup = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    auto start_entire = chrono::high_resolution_clock::now();
    
    // Measure kernel invocation
    printf("START\n");
    start_time = chrono::high_resolution_clock::now();
    int * data;
    cudaMalloc(&data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();

    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

 
    start_time = chrono::high_resolution_clock::now();
    kernel_new<<<GRID_DIM, BLOCK_DIM>>>(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    int * new_data;
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_1<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;
    

    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_2<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_3<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_4<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
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


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_6<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_7<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_8<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_9<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // NEXT KERNEL
    start_time = chrono::high_resolution_clock::now();
    cudaMalloc(&new_data, (sizeof(int) * 60000000));
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    kernel_10<<<GRID_DIM, BLOCK_DIM>>>(new_data, data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_kernel += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    start_time = chrono::high_resolution_clock::now();
    cudaFree(data);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_free += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    data = new_data;


    // Copy back
    start_time = chrono::high_resolution_clock::now();
    int * tmp_result = (int *) malloc(sizeof(int) * 60000000);
    cudaMemcpy(tmp_result, data, sizeof(int) * 60000000, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_transfer += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();


    end_time = chrono::high_resolution_clock::now();
    int time_entire = chrono::duration_cast<chrono::microseconds>(end_time - start_entire).count();

    printf("setup: %f\n", time_setup / 1000.0f);
    printf("alloc: %f\n", time_alloc / 1000.0f);
    printf("kernel: %f\n", time_kernel / 1000.0f);
    printf("transfer: %f\n", time_transfer / 1000.0f);
    printf("free: %f\n", time_free / 1000.f);
    printf("rest: %f\n", (time_entire - time_alloc - time_kernel - time_transfer - time_free) / 1000.0f);

    printf("END\n");  
}
 