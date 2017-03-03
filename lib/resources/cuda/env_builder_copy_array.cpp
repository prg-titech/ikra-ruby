
    void * temp_ptr_/*{field}*/ = /*{host_env}*/->/*{field}*/;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc((void **) &/*{host_env}*/->/*{field}*/, /*{size_bytes}*/));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(/*{host_env}*/->/*{field}*/, temp_ptr_/*{field}*/, /*{size_bytes}*/, cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);
