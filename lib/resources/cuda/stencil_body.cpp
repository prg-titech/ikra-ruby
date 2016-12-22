    /*{result_type}*/ /*{temp_var}*/;

    // Indices for all dimensions
    /*{compute_indices}*/

    if (/*{out_of_bounds_check}*/)
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