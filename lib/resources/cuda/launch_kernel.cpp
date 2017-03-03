    timeStartMeasure();
    /*{kernel_name}*/<<</*{grid_dim}*/, /*{block_dim}*/>>>(/*{arguments}*/);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());
    timeReportMeasure(program_result, kernel);