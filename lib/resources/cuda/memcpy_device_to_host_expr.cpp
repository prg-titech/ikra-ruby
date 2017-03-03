({
    variable_size_array_t device_array = /*{device_array}*/;
    /*{type}*/ * tmp_result = (/*{type}*/ *) malloc(sizeof(/*{type}*/) * device_array.size);

    timeStartMeasure();
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(/*{type}*/) * device_array.size, cudaMemcpyDeviceToHost));
    timeReportMeasure(program_result, transfer_memory);

    variable_size_array_t((void *) tmp_result, device_array.size);
})