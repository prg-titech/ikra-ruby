    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(/*{name}*/));
    timeReportMeasure(program_result, free_memory);
