#include <stdio.h>
#include <assert.h>
#include <chrono>

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
    int l1_hx_res;
    float l1_magnify;
    int l1_hy_res;
    int l1_iter_max;
    int l2_inverted;
};


__device__ int _block_k_1_(environment_t *_env_, int j)
{
    
    float xx;
    int iter;
    float y;
    float x;
    float cy;
    float cx;
    int hy;
    int hx;
    int lex_iter_max = _env_->l1_iter_max;
    int lex_hy_res = _env_->l1_hy_res;
    float lex_magnify = _env_->l1_magnify;
    int lex_hx_res = _env_->l1_hx_res;
    {
        hx = ((j % lex_hx_res));
        hy = ((j / lex_hx_res));
        cx = ((((((((((((float) hx) / ((float) lex_hx_res))) - 0.5)) / lex_magnify)) * 3.0)) - 0.7));
        cy = ((((((((((float) hy) / ((float) lex_hy_res))) - 0.5)) / lex_magnify)) * 3.0));
        x = 0.0;
        y = 0.0;
        for (iter = 0; iter <= lex_iter_max; iter++)
        {
            {
                xx = ((((((x * x)) - ((y * y)))) + cx));
                y = ((((((2.0 * x)) * y)) + cy));
                x = xx;
                if (((((((x * x)) + ((y * y)))) > 100)))
                {
                    break;
                }
            }
        }
        iter--;
        {
            return (iter % 256);
        }
    }
}


__device__ int _block_k_2_(environment_t *_env_, int color)
{
    
    int lex_inverted = _env_->l2_inverted;
    if (((lex_inverted == 1)))
    {
        {
            {
                return (255 - color);
            }
        }
    }
    else
    {
        {
            return color;
        }
    }
}


__global__ void kernel_1(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _block_k_1_(_env_, _tid_));
    }
}


typedef struct result_t {
    int *result;
    int last_error;

    uint64_t time_setup_cuda;
    uint64_t time_prepare_env;
    uint64_t time_kernel;
    uint64_t time_free_memory;
} result_t;

#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return result_var;\
}

#define timeStartMeasure() start_time = chrono::high_resolution_clock::now();

#define timeReportMeasure(result_var, variable_name) \
end_time = chrono::high_resolution_clock::now(); \
result_var->time_##variable_name = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();

extern "C" EXPORT result_t *launch_kernel(environment_t *host_env)
{
    // Variables for measuring time
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;

    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));

    timeStartMeasure();

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        program_result->last_error = -1;
        return program_result;
    }

    checkErrorReturn(program_result, cudaFree(0));

    timeReportMeasure(program_result, setup_cuda);


    /* Prepare environment */
timeStartMeasure();
    /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkErrorReturn(program_result, cudaMalloc(&dev_env, sizeof(environment_t)));
    checkErrorReturn(program_result, cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));

timeReportMeasure(program_result, prepare_env);

    /* Launch all kernels */
timeStartMeasure();
    int * _kernel_result_2;
    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_2, (4 * 40000)));
    int * _kernel_result_2_host = (int *) malloc((4 * 40000));
    kernel_1<<<157, 256>>>(dev_env, 40000, _kernel_result_2);
    checkErrorReturn(program_result, cudaPeekAtLastError());
    checkErrorReturn(program_result, cudaThreadSynchronize());

    checkErrorReturn(program_result, cudaMemcpy(_kernel_result_2_host, _kernel_result_2, (4 * 40000), cudaMemcpyDeviceToHost));

timeReportMeasure(program_result, kernel);

    /* Free device memory */
timeStartMeasure();
    checkErrorReturn(program_result, cudaFree(_kernel_result_2));

timeReportMeasure(program_result, free_memory);

    program_result->result = _kernel_result_2_host;
    return program_result;
}
