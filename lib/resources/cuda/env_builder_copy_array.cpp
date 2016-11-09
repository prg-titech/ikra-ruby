
    void * temp_ptr_/*{field}*/ = /*{host_env}*/->/*{field}*/;
    checkErrorReturn(kernel_result, cudaMalloc((void **) &/*{host_env}*/->/*{field}*/, /*{size_bytes}*/));
    checkErrorReturn(kernel_result, cudaMemcpy(/*{host_env}*/->/*{field}*/, temp_ptr_/*{field}*/, /*{size_bytes}*/, cudaMemcpyHostToDevice));
