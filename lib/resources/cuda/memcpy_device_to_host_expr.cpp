({
    variable_size_array_t device_array = /*{device_array}*/;
    /*{type}*/ * tmp_result = (/*{type}*/ *) malloc(sizeof(/*{type}*/) * device_array.size);
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(/*{type}*/) * device_array.size, cudaMemcpyDeviceToHost));
    variable_size_array_t((void *) tmp_result, device_array.size);
})