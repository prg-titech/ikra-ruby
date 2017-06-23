    timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(/*{name}*/));
    program_result->device_allocations->erase(
        std::remove(
            program_result->device_allocations->begin(),
            program_result->device_allocations->end(),
            /*{name}*/),
        program_result->device_allocations->end());
    timeReportMeasure(program_result, free_memory);
