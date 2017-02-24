#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <vector>

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

/* ----- BEGIN Environment (lexical variables) ----- */
// environment_struct must be defined later
typedef struct environment_struct environment_t;
/* ----- END Environment (lexical variables) ----- */


/* ----- BEGIN Forward declarations ----- */
typedef struct result_t result_t;
/* ----- END Forward declarations ----- */


/* ----- BEGIN Macros ----- */
#define timeStartMeasure() start_time = chrono::high_resolution_clock::now();

#define timeReportMeasure(result_var, variable_name) \
end_time = chrono::high_resolution_clock::now(); \
result_var->time_##variable_name = chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
/* ----- END Macros ----- */

/* ----- BEGIN Structs ----- */
struct fixed_size_array_t {
    void *content;
    int size;

    fixed_size_array_t(void *content_ = NULL, int size_ = 0) : content(content_), size(size_) { }; 

    static const fixed_size_array_t error_return_value;
};

// error_return_value is used in case a host section terminates abnormally
const fixed_size_array_t fixed_size_array_t::error_return_value = 
    fixed_size_array_t(NULL, 0);

/* ----- BEGIN Union Type ----- */
typedef union union_type_value {
    obj_id_t object_id;
    int int_;
    float float_;
    bool bool_;
    void *pointer;
    fixed_size_array_t fixed_size_array;

    __host__ __device__ union_type_value(int value) : int_(value) { };
    __host__ __device__ union_type_value(float value) : float_(value) { };
    __host__ __device__ union_type_value(bool value) : bool_(value) { };
    __host__ __device__ union_type_value(void *value) : pointer(value) { };
    __host__ __device__ union_type_value(fixed_size_array_t value) : fixed_size_array(value) { };

    __host__ __device__ static union_type_value from_object_id(obj_id_t value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_int(int value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_float(float value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_bool(bool value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_pointer(void *value)
    {
        return union_type_value(value);
    }

    __host__ __device__ static union_type_value from_fixed_size_array_t(fixed_size_array_t value)
    {
        return union_type_value(value);
    }
} union_v_t;

typedef struct union_type_struct
{
    class_id_t class_id;
    union_v_t value;

    __host__ __device__ union_type_struct(
        class_id_t class_id_ = 0, union_v_t value_ = union_v_t(0))
        : class_id(class_id_), value(value_) { };

    static const union_type_struct error_return_value;
} union_t;

// error_return_value is used in case a host section terminates abnormally
const union_type_struct union_t::error_return_value = union_type_struct(0, union_v_t(0));
/* ----- END Union Type ----- */

typedef struct result_t {
    union_t result;
    int last_error;

    uint64_t time_setup_cuda;
    uint64_t time_prepare_env;
    uint64_t time_kernel;
    uint64_t time_free_memory;

    // Memory management
    vector<void*> *device_allocations;
} result_t;
/* ----- END Structs ----- */

struct array_command_1 {
    // Ikra::Symbolic::ArrayIndexCommand
    int *result;
    __host__ __device__ array_command_1(int *result = NULL) : result(result) { }
};
struct array_command_2 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_1 *input_0;
    __host__ __device__ array_command_2(int *result = NULL, array_command_1 *input_0 = NULL) : result(result), input_0(input_0) { }
};
struct array_command_5 {
    // Ikra::Symbolic::ArrayInHostSectionCommand
    int *result;
    fixed_size_array_t input_0;
    __host__ __device__ array_command_5(int *result = NULL, fixed_size_array_t input_0 = fixed_size_array_t::error_return_value) : result(result), input_0(input_0) { }
    int size() { return input_0.size; }
};
struct array_command_6 {
    // Ikra::Symbolic::ArrayCombineCommand
    int *result;
    array_command_5 *input_0;
    __host__ __device__ array_command_6(int *result = NULL, array_command_5 *input_0 = NULL) : result(result), input_0(input_0) { }
    int size() { return input_0->size(); }
};
struct environment_struct
{
};

// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_5(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_7(environment_t *_env_, int _num_threads_, int *_result_, int *_array_9_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_9_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_10(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_12(environment_t *_env_, int _num_threads_, int *_result_, int *_array_14_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_14_[_tid_]);
    }
}



// TODO: There should be a better to check if _block_k_2_ is already defined
#ifndef _block_k_2__func
#define _block_k_2__func
__device__ int _block_k_2_(environment_t *_env_, int j)
{
    
    
    {
        return (j + 1);
    }
}

#endif


__global__ void kernel_15(environment_t *_env_, int _num_threads_, int *_result_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_2_(_env_, _tid_);
    }
}



// TODO: There should be a better to check if _block_k_6_ is already defined
#ifndef _block_k_6__func
#define _block_k_6__func
__device__ int _block_k_6_(environment_t *_env_, int k)
{
    
    
    {
        return (k + 1);
    }
}

#endif


__global__ void kernel_17(environment_t *_env_, int _num_threads_, int *_result_, int *_array_19_)
{
    int _tid_ = threadIdx.x + blockIdx.x * blockDim.x;

    if (_tid_ < _num_threads_)
    {

        
        _result_[_tid_] = _block_k_6_(_env_, _array_19_[_tid_]);
    }
}


#undef checkErrorReturn
#define checkErrorReturn(result_var, expr) \
if (result_var->last_error = expr) \
{\
    cudaError_t error = cudaGetLastError();\
    printf("!!! Cuda Failure %s:%d (%i): '%s'\n", __FILE__, __LINE__, expr, cudaGetErrorString(error));\
    cudaDeviceReset();\
    return union_t::error_return_value;\
}

union_t _host_section__(environment_t *host_env, environment_t *dev_env, result_t *program_result)
{
    array_command_2 * input = new array_command_2();
    int i;
    array_command_6 * _ssa_var_a_2;
    union_t _ssa_var_a_1;
    {
        _ssa_var_a_1 = union_t(10, union_v_t::from_pointer((void *) input));
        for (i = 1; i <= (100000 - 1); i++)
        {
            _ssa_var_a_2 = new array_command_6(NULL, ({
                array_command_5 * _polytemp_result_1;
                {
                    union_t _polytemp_expr_2 = ({
                        union_t _polytemp_result_3;
                        {
                            union_t _polytemp_expr_4 = _ssa_var_a_1;
                            switch (_polytemp_expr_4.class_id)
                            {
                                case 10: _polytemp_result_3 = union_t(11, union_v_t::from_fixed_size_array_t(({
                                    // #<Ikra::Symbolic::ArrayCombineCommand:0x000000006905a0>
                                
                                    array_command_2 * cmd = (array_command_2 *) _polytemp_expr_4.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            int * _kernel_result_6;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_6, (sizeof(int) * 511)));
                                    program_result->device_allocations->push_back(_kernel_result_6);
                                    kernel_5<<<2, 256>>>(dev_env, 511, _kernel_result_6);
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                
                                
                                        cmd->result = _kernel_result_6;
                                    }
                                
                                    fixed_size_array_t((void *) cmd->result, 511);
                                }))); break;
                                case 12: _polytemp_result_3 = union_t(13, union_v_t::from_fixed_size_array_t(({
                                    // #<Ikra::Symbolic::ArrayCombineCommand:0x00000000e9a390>: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                                
                                    array_command_6 * cmd = (array_command_6 *) _polytemp_expr_4.value.pointer;
                                
                                    if (cmd->result == 0) {
                                            int * _kernel_result_8;
                                    checkErrorReturn(program_result, cudaMalloc(&_kernel_result_8, (sizeof(int) * cmd->input_0->size())));
                                    program_result->device_allocations->push_back(_kernel_result_8);
                                    kernel_7<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_8, ((int *) cmd->input_0->input_0.content));
                                    checkErrorReturn(program_result, cudaPeekAtLastError());
                                    checkErrorReturn(program_result, cudaThreadSynchronize());
                                
                                
                                        cmd->result = _kernel_result_8;
                                    }
                                
                                    fixed_size_array_t((void *) cmd->result, cmd->size());
                                }))); break;
                            }
                        }
                        _polytemp_result_3;
                    });
                    switch (_polytemp_expr_2.class_id)
                    {
                        case 11: _polytemp_result_1 = new array_command_5(NULL, _polytemp_expr_2.value.fixed_size_array); break;
                        case 13: _polytemp_result_1 = new array_command_5(NULL, _polytemp_expr_2.value.fixed_size_array); break;
                    }
                }
                _polytemp_result_1;
            }));
            _ssa_var_a_1 = union_t(12, union_v_t::from_pointer((void *) _ssa_var_a_2));
        }
        i--;
        return ({
            union_t _polytemp_result_9;
            {
                union_t _polytemp_expr_10 = _ssa_var_a_1;
                switch (_polytemp_expr_10.class_id)
                {
                    case 10: _polytemp_result_9 = union_t(11, union_v_t::from_fixed_size_array_t(({
                        // #<Ikra::Symbolic::ArrayCombineCommand:0x000000006905a0>
                    
                        array_command_2 * cmd = (array_command_2 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_16;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_16, (sizeof(int) * 511)));
                        program_result->device_allocations->push_back(_kernel_result_16);
                        kernel_15<<<2, 256>>>(dev_env, 511, _kernel_result_16);
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_16;
                        }
                    
                        fixed_size_array_t((void *) cmd->result, 511);
                    }))); break;
                    case 12: _polytemp_result_9 = union_t(13, union_v_t::from_fixed_size_array_t(({
                        // #<Ikra::Symbolic::ArrayCombineCommand:0x00000000e9a390>: [SendNode: [SendNode: [SendNode: [LVarReadNode: _ssa_var_a_1].__call__()].to_command()].pmap()]
                    
                        array_command_6 * cmd = (array_command_6 *) _polytemp_expr_10.value.pointer;
                    
                        if (cmd->result == 0) {
                                int * _kernel_result_18;
                        checkErrorReturn(program_result, cudaMalloc(&_kernel_result_18, (sizeof(int) * cmd->input_0->size())));
                        program_result->device_allocations->push_back(_kernel_result_18);
                        kernel_17<<<max((int) ceil(((float) cmd->input_0->size()) / 256), 1), (cmd->input_0->size() >= 256 ? 256 : cmd->input_0->size())>>>(dev_env, cmd->input_0->size(), _kernel_result_18, ((int *) cmd->input_0->input_0.content));
                        checkErrorReturn(program_result, cudaPeekAtLastError());
                        checkErrorReturn(program_result, cudaThreadSynchronize());
                    
                    
                            cmd->result = _kernel_result_18;
                        }
                    
                        fixed_size_array_t((void *) cmd->result, cmd->size());
                    }))); break;
                }
            }
            _polytemp_result_9;
        });
    }
}

#undef checkErrorReturn
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
    // Variables for measuring time
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;

    // CUDA Initialization
    result_t *program_result = (result_t *) malloc(sizeof(result_t));
    program_result->device_allocations = new vector<void*>();

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


    /* Copy back memory and set pointer of result */
    program_result->result = ({
    union_t _polytemp_result_11;
    {
        union_t _polytemp_expr_12 = _host_section__(host_env, dev_env, program_result);
        switch (_polytemp_expr_12.class_id)
        {
            case 11: _polytemp_result_11 = union_t(14, union_v_t::from_fixed_size_array_t(({
                fixed_size_array_t device_array = _polytemp_expr_12.value.fixed_size_array;
                int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);
                checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
                fixed_size_array_t((void *) tmp_result, device_array.size);
            }))); break;
            case 13: _polytemp_result_11 = union_t(15, union_v_t::from_fixed_size_array_t(({
                fixed_size_array_t device_array = _polytemp_expr_12.value.fixed_size_array;
                int * tmp_result = (int *) malloc(sizeof(int) * device_array.size);
                checkErrorReturn(program_result, cudaMemcpy(tmp_result, device_array.content, sizeof(int) * device_array.size, cudaMemcpyDeviceToHost));
                fixed_size_array_t((void *) tmp_result, device_array.size);
            }))); break;
        }
    }
    _polytemp_result_11;
});

    /* Free device memory */
    timeStartMeasure();

    for (
        auto device_ptr = program_result->device_allocations->begin(); 
        device_ptr < program_result->device_allocations->end(); 
        device_ptr++)
    {
        checkErrorReturn(program_result, cudaFree(*device_ptr));
    }

    delete program_result->device_allocations;

    timeReportMeasure(program_result, free_memory);

    return program_result;
}
