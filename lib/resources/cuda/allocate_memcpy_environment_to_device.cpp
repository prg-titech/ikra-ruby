    /* Allocate device environment and copy over struct */
    environment_t */*{dev_env}*/;

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMalloc(&/*{dev_env}*/, sizeof(environment_t)));
    timeReportMeasure(program_result, allocate_memory);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(/*{dev_env}*/, /*{host_env}*/, sizeof(environment_t), cudaMemcpyHostToDevice));
    timeReportMeasure(program_result, transfer_memory);
    