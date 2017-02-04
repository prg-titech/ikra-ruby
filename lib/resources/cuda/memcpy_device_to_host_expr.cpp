({
    fixed_size_array_t</*{type}*/> device_array = /*{device_array}*/;
    /*{type}*/ * tmp_result = (/*{type}*/ *) malloc(sizeof(/*{type}*/) * device_array.size);
    checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(/*{type}*/) * device_array.size, cudaMemcpyDeviceToHost));
    fixed_size_array_t</*{type}*/>(tmp_result, device_array.size);
})