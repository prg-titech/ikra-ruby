#include <stdio.h>
#include <assert.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

using namespace std;


/* ----- BEGIN Shared Library Export ----- */
// taken from http://stackoverflow.com/questions/2164827/explicitly-exporting-shared-library-functions-in-linux

#if defined(_MSC_VER)
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(_GCC)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif
/* ----- END Shared Library Export ----- */

/* ----- BEGIN Class Type ----- */
typedef int obj_id_t;
typedef int class_id_t;
/* ----- END Class Type ----- */

/* ----- BEGIN Union Type ----- */
typedef union union_type_value {
    obj_id_t object_id;
    int int_;
    float float_;
    bool bool_;
} union_v_t;

typedef struct union_type_struct
{
    class_id_t class_id;
    union_v_t value;
} union_t;
/* ----- END Union Type ----- */


/* ----- BEGIN Environment (lexical variables) ----- */
// environment_struct must be defined later
typedef struct environment_struct environment_t;
/* ----- END Environment (lexical variables) ----- */

struct environment_struct
{
    int * b1_base;
    int b1_size;
};
__device__ int _block_k_2_(environment_t *_env_, int value)
{
    
    {
        return (value * value);
    }
}


__global__ void kernel_7(environment_t *_env_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < 10000)
    {
        
        
        _result_[_tid_] = _block_k_2_(_env_, _env_->b1_base[_tid_]);
    }
}


typedef struct result_t {
    int *result;
    int last_error;
} result_t;

#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return result_var;\
}

extern "C" EXPORT result_t *launch_kernel(environment_t *host_env)
{
    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        program_result->last_error = -1;
        return program_result;
    }

    checkErrorReturn(program_result, cudaFree(0));

    /* Prepare environment */

    void * temp_ptr_b1_base = host_env->b1_base;
    checkErrorReturn(program_result, cudaMalloc((void **) &host_env->b1_base, 40000));
    checkErrorReturn(program_result, cudaMemcpy(host_env->b1_base, temp_ptr_b1_base, 40000, cudaMemcpyHostToDevice));
    /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkErrorReturn(program_result, cudaMalloc(&dev_env, sizeof(environment_t)));
    checkErrorReturn(program_result, cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));



    /* Launch all kernels */
    int * _kernel_result_8;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_8, (4 * 10000)));
    int * _kernel_result_8_host = (int *) malloc((4 * 10000));
    kernel_7<<<40, 250>>>(dev_env, _kernel_result_8);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());

    checkErrorReturn(program_result, cudaMemcpy(_kernel_result_8_host, _kernel_result_8, (4 * 10000), cudaMemcpyDeviceToHost));


    /* Free device memory */
    checkErrorReturn(program_result, cudaFree(_kernel_result_8));

    
    program_result->result = _kernel_result_8_host;
    return program_result;
}
