#include <stdio.h>

__global__ void main_kernel(float magnify, int hx_res, int hy_res, int iter_max, int* _result_)
{

    int i = blockIdx.x;
    int hx = (i % hx_res);
    int hy = (i / hx_res);
    float cx = ((((((float) (hx) / (float) (hx_res)) - 0.5) / magnify) * 3.0) - 0.7);
    float cy = (((((float) (hy) / (float) (hy_res)) - 0.5) / magnify) * 3.0);
    float x = 0.0;
    float y = 0.0;
    for (i = 0; i <= iter_max; i++)
    {
        float xx = (((x * x) - (y * y)) + cx);
        y = (((2.0 * x) * y) + cy);
        x = xx;
        if ((((x * x) + (y * y)) > 100))
        {
            i = 101;
            break;
        }
        ;
    }
    ;
    if ((i == 101))
    {
        ;
        _result_[blockIdx.x] = 0;
    }
    else
    {
        ;
        _result_[blockIdx.x] = 1;
    }
    ;
}

int launch_kernel(float magnify, int hx_res, int hy_res, int iter_max)
{
    int * host_result = (int*) malloc(sizeof(int) * 1);
    int * device_result;

    cudaMalloc(&device_result, sizeof(int) * 1);

    dim3 dim_grid(10000, 1, 1);
    dim3 dim_block(100, 1, 1);
    main_kernel<<<dim_grid, dim_block>>>(magnify, hx_res, hy_res, iter_max, device_result);

    cudaThreadSynchronize();
    cudaMemcpy(host_result, device_result, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    cudaFree(device_result);
    printf("RESULT: %i\n", host_result[0]);
    
    return 1;
}
int main() {
    launch_kernel(1, 1, 1, 1);
}