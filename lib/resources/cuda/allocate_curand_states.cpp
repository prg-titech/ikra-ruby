
    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc(&/*{host_env}*/->states, sizeof(curandState_t) * /*{num_threads}*/));
    checkErrorReturn(program_result, cudaMalloc(&/*{host_env}*/->seeds, sizeof(unsigned long long) * /*{num_threads}*/))
    timeReportMeasure(program_result, allocate_memory);
