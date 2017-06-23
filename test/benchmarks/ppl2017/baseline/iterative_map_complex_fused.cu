#include <chrono>
#include <stdio.h>

#define GRID_DIM 58594
#define BLOCK_DIM 1024

using namespace std;

/*

        base = Array.pnew(dimensions: DIMS) do |indices|
            (indices[2]) % 133777
        end

        return Ikra::Symbolic.host_section(base) do |x|
            y = x
            old_data = x.__call__.to_command

            for r in 0...200
                if r % 2 == 0
                    if r % 3 == 0
                        y = y.pmap(with_index: true) do |i, indices|
                            (i + indices[3]) % 77689
                        end
                    else
                        y = y.pmap(with_index: true) do |i, indices|
                            (i + indices[0]) % 11799
                        end
                    end
                else
                    y = y.pmap(with_index: true) do |i, indices|
                        (i + indices[2]) % 1337
                    end

                    y = y.pmap(with_index: true) do |i, indices|
                        (i + indices[2]) % 8888888
                    end
                end

                y = y.pmap(with_index: true) do |i, indices|
                    (i + indices[2]) % 6678
                end

                old_data.free_memory
                old_data = y
            end

            y
        end

*/

__global__ void kernel_new(int *data) {
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    // int indices[] = {idx_0, idx_1, idx_2, idx_3};

    data[_tid_] = idx_2 % 133777;
}

__global__ void kernel_1(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    new_data[_tid_] = (data[_tid_] + idx_3) % 77689;
}

__global__ void kernel_2(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    new_data[_tid_] = (data[_tid_] + idx_0) % 11799;
}

__global__ void kernel_3(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    new_data[_tid_] = (data[_tid_] + idx_2) % 1337;
}

__global__ void kernel_4(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    new_data[_tid_] = (data[_tid_] + idx_2) % 8888888;
}

__global__ void kernel_5(int *new_data, int *data)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 60000000) return;

    int idx_0 = _tid_ / (12*500*500);
    int idx_1 = (_tid_ / (12*500)) % 500;
    int idx_2 = (_tid_ / 12) % 500;
    int idx_3 = (_tid_ / 1) % 12;

    new_data[_tid_] = (data[_tid_] + idx_2) % 6678;
}

int main()
{
    long time_kernel = 0;
    long time_alloc = 0;
    long time_free = 0;
    long time_transfer = 0;

    auto start_time = chrono::high_resolution_clock::now();
    auto end_time = chrono::high_resolution_clock::now();

    // Init
    start_time = chrono::high_resolution_clock::now();
    cudaThreadSynchronize();
    cudaFree(0);
    end_time = chrono::high_resolution_clock::now();;
    long time_setup = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    auto start_entire = chrono::high_resolution_clock::now();

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

    for (int r = 0; r < 200; r++)
    {
        if (r % 2 == 0) {
            if (r % 3 == 0) {
                // KERNEL LAUNCH
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
            }
            else {
                // KERNEL LAUNCH
                start_time = chrono::high_resolution_clock::now();
                int * new_data;
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
            }
        } else {
            // KERNEL LAUNCH
            start_time = chrono::high_resolution_clock::now();
            int * new_data;
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

            // KERNEL LAUNCH
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
        }

        // KERNEL LAUNCH
        start_time = chrono::high_resolution_clock::now();
        int * new_data;
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
    }

    cudaThreadSynchronize();

    // Copy back
    start_time = chrono::high_resolution_clock::now();
    int * tmp_result = (int *) malloc(sizeof(int) * 60000000);
    cudaMemcpy(tmp_result, data, sizeof(int) * 60000000, cudaMemcpyDeviceToHost);
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
