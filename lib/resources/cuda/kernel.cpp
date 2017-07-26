

__global__ void /*{kernel_name}*/(/*{parameters}*/)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < /*{num_threads}*/)
    {
        curand_init(/*{curand_seed}*/, _tid_, 0, _env_->states + _tid_);

/*{execution}*/

        _result_[_tid_] = /*{block_invocation}*/;
    }
}


