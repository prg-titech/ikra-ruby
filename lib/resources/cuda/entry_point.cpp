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

extern "C" EXPORT result_t *launch_kernel(environment_t */*{host_env_var_name}*/)
{
    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        program_result->last_error = -1;
        return program_result;
    }

    checkErrorReturn(program_result, cudaFree(0));

    /* Prepare environment */
/*{prepare_environment}*/


    /* Launch all kernels */
/*{launch_all_kernels}*/

    /* Free device memory */
/*{free_device_memory}*/
    
    program_result->result = /*{host_result_var_name}*/;
    return program_result;
}
