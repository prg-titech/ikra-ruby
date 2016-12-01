

__global__ void /*{kernel_name}*/(/*{parameters}*/)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < /*{num_threads}*/)
    {
/*{execution}*/
        
        _result_[_tid_] = /*{block_invocation}*/;
    }
}


