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

__device__ int _method_singleton_Object_encodeHSBcolor_(environment_t * _env_, obj_id_t _self_, float h, float s, float b)
{
    float c;
    float h_;
    float x;
    union_t r1;
    union_t g1;
    union_t b1;
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
                r1 = ((union_t) {2, {.float_ = c}});
                g1 = ((union_t) {2, {.float_ = x}});
                b1 = ((union_t) {1, {.int_ = 0}});
            }
        }
        else
        {
            if ((h_ < 2))
            {
                {
                    r1 = ((union_t) {2, {.float_ = x}});
                    g1 = ((union_t) {2, {.float_ = c}});
                    b1 = ((union_t) {1, {.int_ = 0}});
                }
            }
            else
            {
                if ((h_ < 3))
                {
                    {
                        r1 = ((union_t) {1, {.int_ = 0}});
                        g1 = ((union_t) {2, {.float_ = c}});
                        b1 = ((union_t) {2, {.float_ = x}});
                    }
                }
                else
                {
                    if ((h_ < 4))
                    {
                        {
                            r1 = ((union_t) {1, {.int_ = 0}});
                            g1 = ((union_t) {2, {.float_ = x}});
                            b1 = ((union_t) {2, {.float_ = c}});
                        }
                    }
                    else
                    {
                        if ((h_ < 5))
                        {
                            {
                                r1 = ((union_t) {2, {.float_ = x}});
                                g1 = ((union_t) {1, {.int_ = 0}});
                                b1 = ((union_t) {2, {.float_ = c}});
                            }
                        }
                        else
                        {
                            {
                                r1 = ((union_t) {2, {.float_ = c}});
                                g1 = ((union_t) {1, {.int_ = 0}});
                                b1 = ((union_t) {2, {.float_ = x}});
                            }
                        }
                    }
                }
            }
        }
        m = (b - (c / 2));
        r = ({
            union_t _polytemp_recv_1 = r1;
            float _polytemp_result_1;
            switch (_polytemp_recv_1.class_id)
            {
                case 2: _polytemp_result_1 = (_polytemp_recv_1.value.float_ + m); break;
                case 1: _polytemp_result_1 = (_polytemp_recv_1.value.int_ + m); break;
            }
            _polytemp_result_1;
        });
        g = ({
            union_t _polytemp_recv_2 = g1;
            float _polytemp_result_2;
            switch (_polytemp_recv_2.class_id)
            {
                case 2: _polytemp_result_2 = (_polytemp_recv_2.value.float_ + m); break;
                case 1: _polytemp_result_2 = (_polytemp_recv_2.value.int_ + m); break;
            }
            _polytemp_result_2;
        });
        b = ({
            union_t _polytemp_recv_3 = b1;
            float _polytemp_result_3;
            switch (_polytemp_recv_3.class_id)
            {
                case 1: _polytemp_result_3 = (_polytemp_recv_3.value.int_ + m); break;
                case 2: _polytemp_result_3 = (_polytemp_recv_3.value.float_ + m); break;
            }
            _polytemp_result_3;
        });
        return (((((int) ((r * 255))) * 65536) + (((int) ((g * 255))) * 256)) + ((int) ((b * 255))));
    }
}
__device__ int _block_k_2_(environment_t *_env_, int index)
{
    int x;
    int lex_hx_res = _env_->l3_hx_res;
    {
        x = ((index % lex_hx_res));
        return _method_singleton_Object_encodeHSBcolor_(_env_, NULL, ((((float) x) / lex_hx_res)), 1.0, 0.5);
    }
}


__device__ int _block_k_1_(environment_t *_env_, int index)
{
    {
        return (255 - ((index % 32)));
    }
}


__device__ int _block_k_3_(environment_t *_env_, int value, int index, int value2)
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
        smaller_dim = [&]{
            if (((lex_hx_res < lex_hy_res)))
            {
                {
                    return lex_hx_res;
                }
            }
            else
            {
                {
                    return lex_hy_res;
                }
            }
        }();
        if (((((((delta_x * delta_x)) + ((delta_y * delta_y)))) < ((((smaller_dim * smaller_dim)) / 5)))))
        {
            return value;
        }
        else
        {
            return value2;
        }
    }
}


__global__ void kernel(environment_t *_env_, int *_result_)
{
    int t_id = threadIdx.x + blockIdx.x * blockDim.x;

    if (t_id < 16000000)
    {
        _result_[t_id] = _block_k_3_(_env_, _block_k_1_(_env_, threadIdx.x + blockIdx.x * blockDim.x), threadIdx.x + blockIdx.x * blockDim.x, _block_k_2_(_env_, threadIdx.x + blockIdx.x * blockDim.x));
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
    
    dim3 dim_grid(62500, 1, 1);
    dim3 dim_block(256, 1, 1);

    for (int i = 0; i < 100; i++) {
        kernel<<<dim_grid, dim_block>>>(dev_env, device_result);
    }

    checkCudaErrors(cudaThreadSynchronize());

    checkCudaErrors(cudaMemcpy(host_result, device_result, sizeof(int) * 16000000, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(dev_env));

    return host_result;
}
