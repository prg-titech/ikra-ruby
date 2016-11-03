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
    int l2_size;
    int * l2_a;
    int * l2_b;
};
__device__ int _block_k_2_(environment_t *_env_, int index)
{
    int i;
    int result;
    int y;
    int x;
    int * lex_b = _env_->l2_b;
    int * lex_a = _env_->l2_a;
    int lex_size = _env_->l2_size;
    {
        x = (((index) % (lex_size)));
        y = (((index) / (lex_size)));
        result = 0;
        for (i = 0; i <= ((lex_size) - (1)); i++)
        result = (((result) + (((((lex_a)[((((((y) * (lex_size)))) + (i)))]) * ((lex_b)[((((((i) * (lex_size)))) + (x)))]))))));
        i--;
        return result;
    }
}


__global__ void kernel(environment_t *_env_, int *_result_)
{
    _result_[threadIdx.x + blockIdx.x * blockDim.x] = _block_k_2_(_env_, threadIdx.x + blockIdx.x * blockDim.x);
}


extern "C" EXPORT int *launch_kernel(environment_t *host_env)
{
    /* Modify host environment to contain device pointers addresses */
    
    void * temp_ptr_l2_a = host_env->l2_a;
    checkCudaErrors(cudaMalloc((void **) &host_env->l2_a, 90000));
    checkCudaErrors(cudaMemcpy(host_env->l2_a, temp_ptr_l2_a, 90000, cudaMemcpyHostToDevice));

    void * temp_ptr_l2_b = host_env->l2_b;
    checkCudaErrors(cudaMalloc((void **) &host_env->l2_b, 90000));
    checkCudaErrors(cudaMemcpy(host_env->l2_b, temp_ptr_l2_b, 90000, cudaMemcpyHostToDevice));


    /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkCudaErrors(cudaMalloc(&dev_env, sizeof(environment_t)));
    checkCudaErrors(cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));

    int *host_result = (int *) malloc(sizeof(int) * 22500);
    int *device_result;
    checkCudaErrors(cudaMalloc(&device_result, sizeof(int) * 22500));
    
    dim3 dim_grid(90, 1, 1);
    dim3 dim_block(250, 1, 1);
    
    kernel<<<dim_grid, dim_block>>>(dev_env, device_result);

    checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaMemcpy(host_result, device_result, sizeof(int) * 22500, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_env));

    return host_result;
}
