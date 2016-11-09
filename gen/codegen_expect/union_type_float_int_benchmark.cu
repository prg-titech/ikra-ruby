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
    int l1_hx_res;
};
__device__ int _method_singleton_Object_encodeHSBcolor_(environment_t * _env_, obj_id_t _self_, float h, int s, float b)
{
    float c;
    float h_;
    float x;
    float r1;
    float g1;
    float b1;
    float m;
    float r;
    float g;
    {
        c = (((1 - fabsf((((2 * b) - 1))))) * s);
        h_ = ((((h - floorf(h))) * 360) / 60);
        x = (c * ((1 - fabsf(((fmodf(h_, ((float) 2)) - 1))))));
        if ((h_ < 1))
        {
            {
                r1 = c;
                g1 = x;
                b1 = 0;
            }
        }
        else
        {
            if ((h_ < 2))
            {
                {
                    r1 = x;
                    g1 = c;
                    b1 = 0;
                }
            }
            else
            {
                if ((h_ < 3))
                {
                    {
                        r1 = 0;
                        g1 = c;
                        b1 = x;
                    }
                }
                else
                {
                    if ((h_ < 4))
                    {
                        {
                            r1 = 0;
                            g1 = x;
                            b1 = c;
                        }
                    }
                    else
                    {
                        if ((h_ < 5))
                        {
                            {
                                r1 = x;
                                g1 = 0;
                                b1 = c;
                            }
                        }
                        else
                        {
                            {
                                r1 = c;
                                g1 = 0;
                                b1 = x;
                            }
                        }
                    }
                }
            }
        }
        m = (b - (c / 2));
        r = (r1 + m);
        g = (g1 + m);
        b = (b1 + m);
        return (((((int) ((r * 255))) * 65536) + (((int) ((g * 255))) * 256)) + ((int) ((b * 255))));
    }
}
__device__ int _block_k_1_(environment_t *_env_, int j)
{
    int hy;
    int hx;
    int lex_hx_res = _env_->l1_hx_res;
    {
        hx = ((j % lex_hx_res));
        hy = ((j / lex_hx_res));
        return _method_singleton_Object_encodeHSBcolor_(_env_, NULL, ((((float) hx) / lex_hx_res)), 1, 0.5);
    }
}


__global__ void kernel(environment_t *_env_, int *_result_)
{
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id < 16000000)
    {
        _result_[t_id] = _block_k_1_(_env_, threadIdx.x + blockIdx.x * blockDim.x);
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
    result_t *kernel_result = (result_t *) malloc(sizeof(result_t));

    cudaError_t cudaStatus = cudaSetDevice(0);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        kernel_result->last_error = -1;
        return kernel_result;
    }

    checkErrorReturn(kernel_result, cudaFree(0));

    /* Modify host environment to contain device pointers addresses */
    

    /* Allocate device environment and copy over struct */
    environment_t *dev_env;
    checkErrorReturn(kernel_result, cudaMalloc(&dev_env, sizeof(environment_t)));
    checkErrorReturn(kernel_result, cudaMemcpy(dev_env, host_env, sizeof(environment_t), cudaMemcpyHostToDevice));

    int *host_result = (int *) malloc(sizeof(int) * 16000000);
    int *device_result;
    checkErrorReturn(kernel_result, cudaMalloc(&device_result, sizeof(int) * 16000000));
    
    dim3 dim_grid(62500, 1, 1);
    dim3 dim_block(256, 1, 1);

    kernel<<<dim_grid, dim_block>>>(dev_env, device_result);

    checkErrorReturn(kernel_result, cudaPeekAtLastError());
    checkErrorReturn(kernel_result, cudaThreadSynchronize());

    checkErrorReturn(kernel_result, cudaMemcpy(host_result, device_result, sizeof(int) * 16000000, cudaMemcpyDeviceToHost));
    checkErrorReturn(kernel_result, cudaFree(dev_env));

    kernel_result->result = host_result;
    return kernel_result;
}
