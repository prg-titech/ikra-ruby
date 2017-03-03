    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(/*{host_name}*/, /*{device_name}*/, /*{bytes}*/, cudaMemcpyDeviceToHost));
    timeReportMeasure(program_result, transfer_memory);
