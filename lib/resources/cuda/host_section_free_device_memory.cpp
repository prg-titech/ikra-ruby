    timeStartMeasure();

    if (/*{name}*/ != cmd->result) {
        // Don't free memory if it is the result. There is already a similar check in
        // program_builder (free all except for last). However, this check is not sufficient in
        // case the same array is reused!

        checkErrorReturn(program_result, cudaFree(/*{name}*/));
        // Remove from list of allocations
        program_result->device_allocations->erase(
            std::remove(
                program_result->device_allocations->begin(),
                program_result->device_allocations->end(),
                /*{name}*/),
            program_result->device_allocations->end());
    }

    timeReportMeasure(program_result, free_memory);
