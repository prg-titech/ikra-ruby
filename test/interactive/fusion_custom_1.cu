#include <stdio.h>

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
    int l3_hx_res;
    int l3_hy_res;
};
__device__ int _block_k_1_(environment_t *_env_, int tid)
{
    return 255 - tid % 32;
}

__device__ int _block_k_3_unfused(environment_t *_env_, int *dependent_computation, int index)
{
    int smaller_dim;
    int delta_y;
    int delta_x;
    int y;
    int x;
    int lex_hy_res = _env_->l3_hy_res;
    int lex_hx_res = _env_->l3_hx_res;
    {
        x = ((index % lex_hx_res));
        y = ((index / lex_hx_res));
        delta_x = ((((lex_hx_res / 2)) - x));
        delta_y = ((((lex_hy_res / 2)) - y));
        if (((((((delta_x * delta_x)) + ((delta_y * delta_y)))) < ((((lex_hy_res * lex_hy_res)) / 5)))))
        {
            // Use temporary array here instead of "value"
            return dependent_computation[index];
        }
        else
        {
            return 0x00ff0000;
        }
    }
}

__global__ void kernel(environment_t *_env_, int *_result_, int * temp_result_)
{
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id < 16000000)
    {
        // First block
        temp_result_[t_id] = _block_k_1_(_env_, threadIdx.x + blockIdx.x * blockDim.x);

        // Second block
        _result_[t_id] = _block_k_3_unfused(_env_, temp_result_, threadIdx.x + blockIdx.x * blockDim.x);
    }
}


extern "C" EXPORT int *launch_kernel(environment_t *host_env)
{
    /* Modify host environment to contain device pointers addresses */


    /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkCudaErrors(cudaMalloc(&dev_env, sizeof(environment_t)));
    checkCudaErrors(cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));

    int *host_result = (int *) malloc(sizeof(int) * 16000000);
    int *device_result;
    checkCudaErrors(cudaMalloc(&device_result, sizeof(int) * 16000000));

    int *device_temp_result;
    checkCudaErrors(cudaMalloc(&device_temp_result, sizeof(int) * 16000000));
    
    dim3 dim_grid(62500, 1, 1);
    dim3 dim_block(256, 1, 1);
    
    for (int i = 0; i < 100; i++) {
        kernel<<<dim_grid, dim_block>>>(dev_env, device_result, device_temp_result);
    }
    
    checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaMemcpy(host_result, device_result, sizeof(int) * 16000000, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_env));

    return host_result;
}
