
    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(/*{host_env}*/->states));
    checkErrorReturn(program_result, cudaFree(/*{host_env}*/->seeds));
    timeReportMeasure(program_result, free_memory);
