
    {
        /*{type}*/ * tmp_result = (/*{type}*/ *) malloc(/*{bytes}*/);

        timeStartMeasure();
        checkErrorReturn(program_result, cudaMemcpy(tmp_result, program_result->result, /*{bytes}*/, cudaMemcpyDeviceToHost));
        timeReportMeasure(program_result, transfer_memory);

        program_result->result = tmp_result;
    }
