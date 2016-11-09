typedef struct result_t {
    /*{result_type}*/ *result;
    int last_error;
} result_t;

#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return result_var;\
}

extern "C" EXPORT result_t *launch_kernel(environment_t */*{host_env}*/)
{
    // CUDA Initialization
    result_t *kernel_result = (result_t *) malloc(sizeof(result_t));

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        kernel_result->last_error = -1;
        return kernel_result;
    }

    checkErrorReturn(kernel_result, cudaFree(0));

    /* Modify host environment to contain device pointers addresses */
    /*{copy_env}*/

    /* Allocate device environment and copy over struct */
    environment_t */*{dev_env}*/;
    checkErrorReturn(kernel_result, cudaMalloc(&/*{dev_env}*/, sizeof(environment_t)));
    checkErrorReturn(kernel_result, cudaMemcpy(/*{dev_env}*/, /*{host_env}*/, sizeof(environment_t), cudaMemcpyHostToDevice));

    /*{result_type}*/ *host_result = (/*{result_type}*/ *) malloc(sizeof(/*{result_type}*/) * /*{result_size}*/);
    /*{result_type}*/ *device_result;
    checkErrorReturn(kernel_result, cudaMalloc(&device_result, sizeof(/*{result_type}*/) * /*{result_size}*/));
    
    dim3 dim_grid(/*{grid_dim[0]}*/, /*{grid_dim[1]}*/, /*{grid_dim[2]}*/);
    dim3 dim_block(/*{block_dim[0]}*/, /*{block_dim[1]}*/, /*{block_dim[2]}*/);

    kernel<<<dim_grid, dim_block>>>(/*{dev_env}*/, device_result);

    checkErrorReturn(kernel_result, cudaPeekAtLastError());
    checkErrorReturn(kernel_result, cudaThreadSynchronize());

    checkErrorReturn(kernel_result, cudaMemcpy(host_result, device_result, sizeof(/*{result_type}*/) * /*{result_size}*/, cudaMemcpyDeviceToHost));
    checkErrorReturn(kernel_result, cudaFree(/*{dev_env}*/));

    kernel_result->result = host_result;
    return kernel_result;
}
