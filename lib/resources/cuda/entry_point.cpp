#undef checkErrorReturn
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
    // Variables for measuring time
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;

    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));
    program_result->device_allocations = new vector<void*>();

    timeStartMeasure();

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        program_result->last_error = -1;
        return program_result;
    }

    checkErrorReturn(program_result, cudaFree(0));

    timeReportMeasure(program_result, setup_cuda);


    /* Prepare environment */
    timeStartMeasure();
    /*{prepare_environment}*/
    timeReportMeasure(program_result, prepare_env);

    /* Launch all kernels */
    timeStartMeasure();
    /*{launch_all_kernels}*/
    timeReportMeasure(program_result, kernel);

    /* Copy over result to the host */
    program_result->result = /*{host_result_array}*/;

    /* Free device memory */
    timeStartMeasure();
    /*{free_device_memory}*/
    timeReportMeasure(program_result, free_memory);

    delete program_result->device_allocations;
    
    return program_result;
}
