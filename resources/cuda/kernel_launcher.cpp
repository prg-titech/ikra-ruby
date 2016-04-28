extern "C" EXPORT bool *launch_kernel(/*{parameters_definition}*/)
{
    printf("kernel launched\n");
    environment_t *device_env;
    cutilSafeCall(cudaMalloc(&device_env, sizeof(environment_t)));
    cutilSafeCall(cudaMemcpy(device_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));
    
    /*{allocate_copy_objects}*/
    
    dim3 dim_grid(/*{grid_dim[0]}*/, /*{grid_dim[1]}*/, /*{grid_dim[2]}*/);
    dim3 dim_block(/*{block_dim[0]}*/, /*{block_dim[1]}*/, /*{block_dim[2]}*/);
    
    kernel<<<dim_grid, dim_block>>>(/*{kernel_launch_arguments}*/;
    cutilCheckMsg("Kernel execution failed");

    cutilSafeCall(cudaThreadSynchronize());

    /*{copy_back_free_objects}*/

    cutilSafeCall(cudaFree(device_env));

    return true;
}
