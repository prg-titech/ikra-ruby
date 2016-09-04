

__global__ void kernel(environment_t */*{env_identifier}*/, /*{result_type}*/ *_result_)
{
    _result_[threadIdx.x + blockIdx.x * blockDim.x] = /*{block_invocation}*/;
}


