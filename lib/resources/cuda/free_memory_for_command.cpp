({
    /*{type}*/ cmd_to_free = /*{receiver}*/;

    timeStartMeasure();
    bool freed_memory = false;

    if (cmd_to_free->result != 0) {
        checkErrorReturn(program_result, cudaFree(cmd_to_free->result));;

        // Remove from list of allocations
        program_result->device_allocations->erase(
            std::remove(
                program_result->device_allocations->begin(),
                program_result->device_allocations->end(),
                cmd_to_free->result),
            program_result->device_allocations->end());

        freed_memory = true;
    }

    timeReportMeasure(program_result, free_memory);
    
    freed_memory;
})