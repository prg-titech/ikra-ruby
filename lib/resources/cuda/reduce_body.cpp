        int thread_idx = threadIdx.x;

        // Single result of this block
        /*{type}*/ /*{temp_result}*/;

        int num_args = 2 * /*{block_size}*/;
        if (blockIdx.x == gridDim.x - 1)
        {
            // Processing the last block, which might be odd (number of elements to reduce).
            // Other blocks cannot be "odd", because every block reduces 2*block_size many elements.

            // Number of elements to reduce in the last block
            num_args = ((2 * /*{num_threads}*/ - 1) % (2 * /*{block_size}*/)) + (/*{odd}*/ ? 0 : 1);
        }

        if (num_args == 1)
        {
            /*{temp_result}*/ = /*{previous_result}*/[_tid_];
        }
        else if (num_args == 2)
        {
            /*{temp_result}*/ = /*{block_name}*/(/*{arguments}*/, /*{previous_result}*/[_tid_], /*{previous_result}*/[_tid_ + /*{num_threads}*/]);
        }
        else
        {
            // Allocate block_size many slots to contain the result of up to block_size many reductions, i.e.,
            // this array contains the reduction of (up to) 2*block_size many elements.
            __shared__ /*{type}*/ sdata[/*{block_size}*/];

            /*{odd}*/ = num_args % 2 == 1;

            // --- FIRST REDUCTION ---  Load from global memory
            // Number of elements after the first reduction
            num_args = num_args / 2 + num_args % 2;

            if (thread_idx == num_args - 1 && /*{odd}*/)
            {
                // This is the last thread, and it should reduce only one element.
                sdata[thread_idx] = /*{previous_result}*/[_tid_];
            }
            else
            {
                sdata[thread_idx] = /*{block_name}*/(/*{arguments}*/, /*{previous_result}*/[_tid_], /*{previous_result}*/[_tid_ + /*{num_threads}*/]);
            }

            __syncthreads();


            // --- SUBSEQUENT REDUCTION ---  Read from shared memory only
            /*{odd}*/ = num_args % 2 == 1;

            for (
                num_args = num_args / 2 + num_args % 2;             // Number of elements after this reduction
                num_args > 1;                                       // ... as long as there's at least 3 elements left
                num_args = num_args / 2 + num_args % 2) {

                if (thread_idx < num_args) {
                    // This thread has work to do...

                    if (thread_idx != num_args - 1 || !/*{odd}*/)
                    {
                        sdata[thread_idx] = /*{block_name}*/(/*{arguments}*/, sdata[thread_idx], sdata[thread_idx + num_args]);
                    }
                    else
                    {
                        // This is the last element and it is odd, do nothing
                    }
                }

                __syncthreads();

                /*{odd}*/ = num_args % 2 == 1;
            }

            if (thread_idx == 0)
            {
                // Last thread returns result
                /*{temp_result}*/ = /*{block_name}*/(/*{arguments}*/, sdata[0], sdata[1]);
            }
        }

        // Write result to different position
        _tid_ = blockIdx.x;

        if (thread_idx != 0) {
            // Only one thread should report the result
            return;
        }