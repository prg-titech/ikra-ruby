#include <chrono>
#include <stdio.h>

#define GRID_DIM 533   
#define BLOCK_DIM 1024

using namespace std;

__global__ void kernel_stencil(float *new_data, float *data, float *param_a, float *param_b, float *param_c, float *param_wrk, float *param_bnd) {

    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ >= 129 * 65 * 65) return;

    int idx_0 =_tid_ / 65 / 65;
    int idx_1 = (_tid_ / 65) % 65;
    int idx_2 = (_tid_ / 1) % 65;

    if (idx_0 - 1 < 0 || idx_0 + 1 >= 129) { new_data[_tid_] = 0.0; return; }
    if (idx_1 - 1 < 0 || idx_2 + 1 >= 65) { new_data[_tid_] = 0.0; return; }
    if (idx_1 - 1 < 0 || idx_2 + 1 >= 65) { new_data[_tid_] = 0.0; return; }

    float v000 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2)];
    float v100 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2)];
    float v010 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
    float v001 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
    float v110 = data[(idx_0 + 1) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
    float v120 = data[(idx_0 + 1) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
    float v210 = data[(idx_0 - 1) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2)];
    float v220 = data[(idx_0 - 1) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
    float v011 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2 + 1)];
    float v021 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2 + 1)];
    float v012 = data[(idx_0) * 65 * 65 + (idx_1 + 1) * 65 + (idx_2 - 1)];
    float v022 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2 - 1)];
    float v101 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
    float v201 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2 + 1)];
    float v102 = data[(idx_0 + 1) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];
    float v202 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];
    float v200 = data[(idx_0 - 1) * 65 * 65 + (idx_1) * 65 + (idx_2)];
    float v020 = data[(idx_0) * 65 * 65 + (idx_1 - 1) * 65 + (idx_2)];
    float v002 = data[(idx_0) * 65 * 65 + (idx_1) * 65 + (idx_2 - 1)];

    new_data[_tid_] = 
        v000 + 0.8 * (((
            param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 0] * v100 + 
            param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 1] * v010 + 
            param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 2] * v001 + 
            param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 0] * 
                (v110 - v120 - v210 + v220) + 
            param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 1] * 
                (v011 - v021 - v012 + v022) + 
            param_b[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 2] * 
                (v101 - v201 - v102 + v202) +
            param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 0] * v200 + 
            param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 1] * v020 + 
            param_c[65 * 65 * 3 * idx_0 + 65 * 3 * idx_1 + 3 * idx_2 + 2] * v002 + 
            param_wrk[65 * 65 * idx_0 + 65 * idx_1 + idx_2]) * 
            param_a[65 * 65 * 4 * idx_0 + 65 * 4 * idx_1 + 4 * idx_2 + 3] - 
            v000) * param_bnd[65 * 65 * idx_0 + 65 * idx_1 + idx_2]);
}

int main()
{

    long time_kernel = 0;
    long time_alloc = 0;
    long time_free = 0;
    long time_transfer = 0;

    // Generate data
    float *p_host = new float[129 * 65 * 65];
    for (int i = 0; i < 129 * 65 * 65; i++) {
        int k = i % 65;
        p_host[i] = (k * k) / (65.0 - 1) / (65.0 - 1);
    }

    float *a_host = new float[129 * 65 * 65 * 4];
    for (int i = 0; i < 129 * 65 * 65 * 4; i++) {
        if (i % 4 == 3) {
            a_host[i] = 1.0 / 6.0;
        } else {
            a_host[i] = 0.0;
        }
    }
    
    float *b_host = new float[129 * 65 * 65 * 3];
    for (int i = 0; i < 129 * 65 * 65 * 3; i++) {
        b_host[i] = 0.0;
    }

    float *c_host = new float[129 * 65 * 65 * 3];
    for (int i = 0; i < 129 * 65 * 65 * 3; i++) {
        c_host[i] = 0.0;
    }

    float *bnd_host = new float[129 * 65 * 65];
    for (int i = 0; i < 129 * 65 * 65; i++) {
        bnd_host[i] = 0.0;
    }

    float *wrk_host = new float[129 * 65 * 65];
    for (int i = 0; i < 129 * 65 * 65; i++) {
        wrk_host[i] = 0.0;
    }

    // Init
    auto start_time = chrono::high_resolution_clock::now();
    cudaThreadSynchronize();
    cudaFree(0);
    auto end_time = chrono::high_resolution_clock::now();;
    long time_setup = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    printf("START\n");
    auto start_entire = chrono::high_resolution_clock::now();

    // Measure kernel invocation
    start_time = chrono::high_resolution_clock::now();
    end_time = chrono::high_resolution_clock::now();
    long loop_time_elapsed;

    // Allocate memory
    float * p_dev;
    cudaMalloc(&p_dev, (sizeof(float) * 129 * 65 * 65));

    float * a_dev;
    cudaMalloc(&a_dev, (sizeof(float) * 129 * 65 * 65 * 4));

    float * b_dev;
    cudaMalloc(&b_dev, (sizeof(float) * 129 * 65 * 65 * 3));

    float * c_dev;
    cudaMalloc(&c_dev, (sizeof(float) * 129 * 65 * 65 * 3));

    float * bnd_dev;
    cudaMalloc(&bnd_dev, (sizeof(float) * 129 * 65 * 65));

    float * wrk_dev;
    cudaMalloc(&wrk_dev, (sizeof(float) * 129 * 65 * 65));
    cudaThreadSynchronize();

    end_time = chrono::high_resolution_clock::now();
    time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

 
    // Transfer initial dataset
    start_time = chrono::high_resolution_clock::now();
    cudaMemcpy(p_dev, p_host, sizeof(float) * 129 * 65 * 65, cudaMemcpyHostToDevice);
    cudaMemcpy(a_dev, a_host, sizeof(float) * 129 * 65 * 65 * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_host, sizeof(float) * 129 * 65 * 65 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c_host, sizeof(float) * 129 * 65 * 65 * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(bnd_dev, bnd_host, sizeof(float) * 129 * 65 * 65, cudaMemcpyHostToDevice);
    cudaMemcpy(wrk_dev, wrk_host, sizeof(float) * 129 * 65 * 65, cudaMemcpyHostToDevice);
    cudaThreadSynchronize();
    end_time = chrono::high_resolution_clock::now();
    time_transfer += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

    float *data = p_dev;

    for (int r = 0; r < 1000; r++)
    {
        start_time = chrono::high_resolution_clock::now();
        float * new_data;
        cudaMalloc(&new_data, (sizeof(float) * 129 * 65 * 65));
        cudaThreadSynchronize();
        end_time = chrono::high_resolution_clock::now();
        time_alloc += chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

        start_time = chrono::high_resolution_clock::now();
        kernel_stencil<<<GRID_DIM, BLOCK_DIM>>>(new_data, data, a_dev, b_dev, c_dev, wrk_dev, bnd_dev);
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
    float * tmp_result = (float *) malloc(sizeof(float) * 129 * 65 * 65);
    cudaMemcpy(tmp_result, data, sizeof(float) * 129 * 65 * 65, cudaMemcpyDeviceToHost);
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
