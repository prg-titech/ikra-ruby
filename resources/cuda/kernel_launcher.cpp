extern "C" EXPORT /*{result_type}*/ *launch_kernel(environment_t *host_env)
{
    printf("kernel launched\n");
    environment_t *device_env;
    cutilSafeCall(cudaMalloc(&device_env, sizeof(environment_t)));
    cutilSafeCall(cudaMemcpy(device_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));
    
    /*{result_type}*/ *host_result = (/*{result_type}*/ *) malloc(sizeof(/*{result_type}*/) * /*{result_size}*/);
    /*{result_type}*/ *device_result;
    cutilSafeCall(cudaMalloc(&device_result, sizeof(/*{result_type}*/) * /*{result_size}*/));
    
    dim3 dim_grid(/*{grid_dim[0]}*/, /*{grid_dim[1]}*/, /*{grid_dim[2]}*/);
    dim3 dim_block(/*{block_dim[0]}*/, /*{block_dim[1]}*/, /*{block_dim[2]}*/);
    
    kernel<<<dim_grid, dim_block>>>(device_env, device_result);
    cutilCheckMsg("Kernel execution failed");

    cutilSafeCall(cudaThreadSynchronize());

    cutilSafeCall(cudaMemcpy(host_result, device_result, sizeof(/*{result_type}*/) * /*{result_type}*/, cudaMemcpyDeviceToHost));
    cutilSafeCall(cudaFree(device_env));

    return host_result;
}
