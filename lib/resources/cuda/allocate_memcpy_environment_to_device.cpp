    /* Allocate device environment and copy over struct */
    environment_t */*{dev_env}*/;
    checkErrorReturn(program_result, cudaMalloc(&/*{dev_env}*/, sizeof(environment_t)));
    checkErrorReturn(program_result, cudaMemcpy(/*{dev_env}*/, /*{host_env}*/, sizeof(environment_t), cudaMemcpyHostToDevice));
