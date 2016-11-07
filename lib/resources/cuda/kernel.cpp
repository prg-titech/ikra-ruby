

__global__ void kernel(environment_t */*{env_identifier}*/, /*{result_type}*/ *_result_)
{
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id < /*{result_size}*/)
    {
        _result_[t_id] = /*{block_invocation}*/;
    }
}


