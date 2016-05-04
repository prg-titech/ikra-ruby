
    void * temp_ptr_/*{field}*/ = /*{host_env}*/->/*{field}*/;
    checkCudaErrors(cudaMalloc((void **) &/*{host_env}*/->/*{field}*/, /*{size_bytes}*/));
    checkCudaErrors(cudaMemcpy(/*{host_env}*/->/*{field}*/, temp_ptr_/*{field}*/, /*{size_bytes}*/, cudaMemcpyHostToDevice));
