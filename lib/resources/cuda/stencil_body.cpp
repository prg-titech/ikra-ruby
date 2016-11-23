    /*{result_type}*/ /*{temp_var}*/;

    if (/*{min_offset}*/ + /*{thread_id}*/ >= 0 
        && /*{max_offset}*/ + /*{thread_id}*/ <= /*{input_size}*/)
    {
        // All value indices within bounds
        /*{execution}*/
        /*{temp_var}*/ = /*{stencil_computation}*/;
    }
    else
    {
        // At least one index is out of bounds
        /*{temp_var}*/ = /*{out_of_bounds_fallback}*/;
    }