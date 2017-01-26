
    {
        /*{type}*/ * tmp_result = (/*{type}*/ *) malloc(/*{bytes}*/);
        checkErrorReturn(program_result, cudaMemcpy(tmp_result, program_result->result, /*{bytes}*/, cudaMemcpyDeviceToHost));
        program_result->result = tmp_result;
    }
